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

RIGHT_HAND_NAME_CANDIDATES = (
    "right_hand",
    "right_wrist",
    "r_hand",
    "r_wrist",
    "rightHand",
    "rightWrist",
    "R_Hand",
    "R_Wrist",
)


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


def d455_pose():
    tripod_r = rpy_matrix(TRIPOD_RPY)
    sensor_r = tripod_r @ rpy_matrix(SENSOR_RPY)
    sensor_pos = TRIPOD_POS + tripod_r @ SENSOR_POS
    return sensor_pos, sensor_r


def project_world(point_world, camera_pos, camera_rot):
    local = (np.asarray(point_world, dtype=float) - camera_pos) @ camera_rot
    depth = float(local[0])
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    u = WIDTH * 0.5 - focal * local[1] / max(depth, 1e-6)
    v = HEIGHT * 0.5 - focal * local[2] / max(depth, 1e-6)
    return np.array([u, v], dtype=float), depth, local


def camera_to_world(point_camera, camera_pos, camera_rot):
    return camera_pos + np.asarray(point_camera, dtype=float) @ camera_rot.T


def confidence_for_pixel(pixel, depth):
    u, v = pixel
    if not (0.0 <= u < WIDTH and 0.0 <= v < HEIGHT and depth > 0.1):
        return 0.0
    center_margin = min(u, WIDTH - 1 - u, v, HEIGHT - 1 - v)
    return float(np.clip(center_margin / 120.0, 0.2, 1.0))


def read_input(path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise RuntimeError(f"Unsupported HMR input format: {path}")


def get_field(data, names):
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]
    return None


def normalize_joint_names(raw):
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return [str(item) for item in raw]


def choose_joint_index(joint_names, joint_index):
    if joint_index is not None:
        return joint_index
    if joint_names:
        lowered = {name.lower(): idx for idx, name in enumerate(joint_names)}
        for candidate in RIGHT_HAND_NAME_CANDIDATES:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
    raise RuntimeError(
        "Could not infer right-hand joint index. Pass --joint-index or provide joint_names "
        f"containing one of: {', '.join(RIGHT_HAND_NAME_CANDIDATES)}"
    )


def extract_right_hand_points(data, joint_index):
    if isinstance(data, dict) and data.get("schema") == "smplx_d455_reconstructed_right_hand_trajectory/v1":
        points = [sample["right_hand_world"] for sample in data["samples"]]
        times = [float(sample["time"]) for sample in data["samples"]]
        phases = [sample.get("phase", "hmr") for sample in data["samples"]]
        return np.asarray(points, dtype=float), times, phases, "world", float(data.get("fps", 15.0))

    joints = get_field(
        data,
        (
            "joints_world",
            "smplx_joints_world",
            "body_joints_world",
            "joints3d_world",
            "joints_camera",
            "smplx_joints_camera",
            "body_joints_camera",
            "joints3d_camera",
            "joints",
            "smplx_joints",
            "body_joints",
            "joints3d",
        ),
    )
    if joints is None:
        raise RuntimeError("Input must contain joints_world/joints_camera/joints or the target retarget schema.")
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise RuntimeError(f"Expected joints shape [frames, joints, 3], got {joints.shape}")

    joint_names = normalize_joint_names(get_field(data, ("joint_names", "joints_names", "names")))
    right_idx = choose_joint_index(joint_names, joint_index)
    if not 0 <= right_idx < joints.shape[1]:
        raise RuntimeError(f"Right-hand joint index {right_idx} is outside joint count {joints.shape[1]}")

    points = joints[:, right_idx, :]
    times = get_field(data, ("times", "timestamps", "time"))
    if times is not None:
        times = [float(v) for v in np.asarray(times).reshape(-1).tolist()]
    phases = get_field(data, ("phases", "phase"))
    if phases is not None:
        phases = [str(v) for v in np.asarray(phases).reshape(-1).tolist()]
    fps = float(get_field(data, ("fps", "frame_rate")) or 15.0)

    source_frame = "camera"
    for name in ("joints_world", "smplx_joints_world", "body_joints_world", "joints3d_world"):
        if isinstance(data, dict) and name in data:
            source_frame = "world"
            break
    return points, times, phases, source_frame, fps


def build_payload(points, times, phases, fps, source_frame, position_scale):
    camera_pos, camera_rot = d455_pose()
    samples = []
    for idx, point in enumerate(points):
        point = np.asarray(point, dtype=float) * position_scale
        if source_frame == "camera":
            world = camera_to_world(point, camera_pos, camera_rot)
        elif source_frame == "world":
            world = point
        else:
            raise RuntimeError(f"Unsupported source frame: {source_frame}")

        pixel, depth, camera_point = project_world(world, camera_pos, camera_rot)
        sample_time = times[idx] if times and idx < len(times) else idx / fps
        phase = phases[idx] if phases and idx < len(phases) else "hmr"
        samples.append(
            {
                "time": round(float(sample_time), 4),
                "phase": phase,
                "right_hand_world": [round(float(v), 6) for v in world],
                "right_hand_camera": [round(float(v), 6) for v in camera_point],
                "pixel": [round(float(v), 2) for v in pixel],
                "depth_m": round(float(depth), 4),
                "confidence": round(confidence_for_pixel(pixel, depth), 4),
            }
        )

    duration = samples[-1]["time"] if samples else 0.0
    return {
        "schema": "smplx_d455_reconstructed_right_hand_trajectory/v1",
        "frame": "world",
        "source": "external_rgbd_hmr_or_smplifyx_adapter",
        "seconds": duration,
        "fps": fps,
        "camera": {
            "topic_rgb": "/tripod_d455/depth/image",
            "topic_depth": "/tripod_d455/depth/depth_image",
            "width": WIDTH,
            "height": HEIGHT,
            "hfov": HFOV,
            "position_world": [round(float(v), 6) for v in camera_pos],
        },
        "diagnostics": {
            "samples": len(samples),
            "input_frame": source_frame,
            "note": "Generated from external RGB-D HMR / SMPLify-X joints for xArm7 retargeting.",
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert RGB-D HMR / SMPLify-X joints into the xArm7 retarget trajectory JSON contract."
    )
    parser.add_argument("--input", required=True, help="HMR / SMPLify-X .json or .npz output.")
    parser.add_argument("--out", required=True, help="Output smplx_d455_reconstructed_right_hand.json path.")
    parser.add_argument("--joint-index", type=int, help="Right hand/wrist joint index if joint_names are unavailable.")
    parser.add_argument(
        "--input-frame",
        choices=["auto", "world", "camera"],
        default="auto",
        help="Coordinate frame for input joints. auto infers from key name.",
    )
    parser.add_argument("--fps", type=float, help="Override FPS when the input has no timestamps.")
    parser.add_argument("--position-scale", type=float, default=1.0, help="Scale input coordinates into meters.")
    args = parser.parse_args()

    data = read_input(Path(args.input))
    points, times, phases, source_frame, inferred_fps = extract_right_hand_points(data, args.joint_index)
    fps = float(args.fps or inferred_fps)
    if args.input_frame != "auto":
        source_frame = args.input_frame
    if times is not None and len(times) != len(points):
        raise RuntimeError(f"times length {len(times)} does not match frame count {len(points)}")
    payload = build_payload(points, times, phases, fps, source_frame, args.position_scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"converted_samples={len(payload['samples'])} source_frame={source_frame} "
        f"fps={fps:.3f} output={out}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
