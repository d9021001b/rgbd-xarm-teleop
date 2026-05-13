#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


WIDTH = 848
HEIGHT = 480
HFOV = 2.05

TRIPOD_POS = np.array([-1.20, -3.10, 0.0], dtype=float)
TRIPOD_RPY = np.array([0.0, 0.0, 1.12], dtype=float)
SENSOR_POS = np.array([0.08, 0.0, 1.44], dtype=float)
SENSOR_RPY = np.array([0.0, 0.12, 0.0], dtype=float)

OBSERVED_RIGHT_HAND_WORLD = {
    "ready": np.array([0.430, -1.432, 1.155], dtype=float),
    "hover": np.array([0.320, -0.915, 1.203], dtype=float),
    "grasp": np.array([0.314, -0.932, 1.147], dtype=float),
    "lift": np.array([0.306, -0.883, 1.351], dtype=float),
}


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_matrix(rpy):
    return rz(float(rpy[2])) @ ry(float(rpy[1])) @ rx(float(rpy[0]))


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def d455_pose():
    tripod_r = rpy_matrix(TRIPOD_RPY)
    sensor_r = tripod_r @ rpy_matrix(SENSOR_RPY)
    sensor_pos = TRIPOD_POS + tripod_r @ SENSOR_POS
    return sensor_pos, sensor_r


def observed_phase_target(t):
    ready = OBSERVED_RIGHT_HAND_WORLD["ready"]
    hover = OBSERVED_RIGHT_HAND_WORLD["hover"]
    grasp = OBSERVED_RIGHT_HAND_WORLD["grasp"]
    lift = OBSERVED_RIGHT_HAND_WORLD["lift"]
    if t < 1.0:
        return ready, "ready"
    if t < 4.8:
        alpha = smoothstep((t - 1.0) / 3.8)
        return ready * (1.0 - alpha) + hover * alpha, "over-table-approach"
    if t < 6.4:
        alpha = smoothstep((t - 4.8) / 1.6)
        return hover * (1.0 - alpha) + grasp * alpha, "descend"
    if t < 7.4:
        return grasp, "grasp"
    alpha = smoothstep((t - 7.4) / 2.6)
    return grasp * (1.0 - alpha) + lift * alpha, "lift"


def project_world(point_world, camera_pos, camera_rot):
    local = (np.asarray(point_world, dtype=float) - camera_pos) @ camera_rot
    depth = float(local[0])
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    u = WIDTH * 0.5 - focal * local[1] / max(depth, 1e-6)
    v = HEIGHT * 0.5 - focal * local[2] / max(depth, 1e-6)
    return np.array([u, v], dtype=float), depth, local


def backproject_to_world(pixel, depth, camera_pos, camera_rot):
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    u, v = pixel
    local = np.array(
        [
            depth,
            -(u - WIDTH * 0.5) * depth / focal,
            -(v - HEIGHT * 0.5) * depth / focal,
        ],
        dtype=float,
    )
    return camera_pos + local @ camera_rot.T, local


def confidence_for_pixel(pixel, depth):
    u, v = pixel
    in_frame = 0.0 <= u < WIDTH and 0.0 <= v < HEIGHT and depth > 0.1
    if not in_frame:
        return 0.0
    center_margin = min(u, WIDTH - 1 - u, v, HEIGHT - 1 - v)
    return float(np.clip(center_margin / 120.0, 0.2, 1.0))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the D455-reconstructed right-hand trajectory consumed by xArm7 retargeting. "
            "This is the simulator reconstruction boundary: it projects the observed SMPL-X hand "
            "through the D455 model, quantizes the RGB-D observation, then back-projects it to world coordinates."
        )
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pixel-noise", type=float, default=0.35)
    parser.add_argument("--depth-noise", type=float, default=0.002)
    args = parser.parse_args()

    camera_pos, camera_rot = d455_pose()
    rng = np.random.default_rng(455)
    frame_count = int(round(args.seconds * args.fps))
    samples = []
    max_error = 0.0
    for frame in range(frame_count + 1):
        t = min(args.seconds, frame / args.fps)
        source_world, phase = observed_phase_target(t)
        pixel, depth, camera_point = project_world(source_world, camera_pos, camera_rot)
        measured_pixel = np.round(pixel + rng.normal(0.0, args.pixel_noise, size=2), 2)
        measured_depth = round(float(depth + rng.normal(0.0, args.depth_noise)), 4)
        reconstructed_world, reconstructed_camera = backproject_to_world(
            measured_pixel, measured_depth, camera_pos, camera_rot
        )
        error = float(np.linalg.norm(reconstructed_world - source_world))
        max_error = max(max_error, error)
        samples.append(
            {
                "time": round(t, 4),
                "phase": phase,
                "right_hand_world": [round(float(v), 6) for v in reconstructed_world],
                "right_hand_camera": [round(float(v), 6) for v in reconstructed_camera],
                "pixel": [round(float(v), 2) for v in measured_pixel],
                "depth_m": measured_depth,
                "confidence": round(confidence_for_pixel(measured_pixel, measured_depth), 4),
                "d455_reprojection_error_m": round(error, 6),
            }
        )

    payload = {
        "schema": "smplx_d455_reconstructed_right_hand_trajectory/v1",
        "frame": "world",
        "source": "gazebo_d455_rgbd_reconstruction_boundary",
        "seconds": args.seconds,
        "fps": args.fps,
        "camera": {
            "topic_rgb": "/tripod_d455/depth/image",
            "topic_depth": "/tripod_d455/depth/depth_image",
            "width": WIDTH,
            "height": HEIGHT,
            "hfov": HFOV,
            "position_world": [round(float(v), 6) for v in camera_pos],
        },
        "diagnostics": {
            "max_reconstruction_error_m": round(max_error, 6),
            "note": (
                "Retargeting consumes only right_hand_world from this file. "
                "Replace this simulator boundary with RGB-D HMR/SMPLify-X output when a real human video is available."
            ),
        },
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"reconstructed_samples={len(samples)} max_error={max_error:.4f} output={out}")


if __name__ == "__main__":
    raise SystemExit(main())
