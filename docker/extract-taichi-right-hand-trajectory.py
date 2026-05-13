#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROBOT_WORLD_POS = np.array([-0.2, -0.5, 1.021], dtype=float)
ROBOT_WORLD_YAW = 1.571
CUP_WORLD = np.array([0.34, -0.93, 1.065], dtype=float)
TCP_GRASP_OFFSET_WORLD = np.array([0.0, 0.0, 0.105], dtype=float)


def load_config(path):
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def world_to_base(point_world, robot_world_pos=ROBOT_WORLD_POS, yaw=ROBOT_WORLD_YAW):
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return rotation @ (np.asarray(point_world, dtype=float) - robot_world_pos)


def base_to_world(point_base, robot_world_pos=ROBOT_WORLD_POS, yaw=ROBOT_WORLD_YAW):
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return robot_world_pos + rotation @ np.asarray(point_base, dtype=float)


def moving_average(values, window):
    if window <= 1 or len(values) == 0:
        return values
    window = int(window)
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    out = np.zeros_like(values)
    for idx in range(values.shape[1]):
        out[:, idx] = np.convolve(padded[:, idx], kernel, mode="valid")
    return out


def interpolate_missing(points, confidences, min_confidence):
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points[:, 0]) & (np.asarray(confidences) >= min_confidence)
    if valid.sum() < 2:
        raise RuntimeError("Not enough confident right-hand detections in the selected clip.")
    xs = np.arange(len(points))
    filled = points.copy()
    for axis in range(points.shape[1]):
        filled[:, axis] = np.interp(xs, xs[valid], points[valid, axis])
    return filled, valid


def phase_for_time(t, duration):
    value = t / max(1e-6, duration)
    if value < 0.18:
        return "taichi-prepare"
    if value < 0.45:
        return "taichi-reach"
    if value < 0.72:
        return "taichi-sweep"
    return "taichi-return"


def draw_overlay(frame, wrist_px, elbow_px, shoulder_px, color=(0, 80, 255)):
    for point, radius, point_color in (
        (shoulder_px, 5, (255, 180, 0)),
        (elbow_px, 5, (80, 220, 255)),
        (wrist_px, 7, color),
    ):
        if point is not None:
            cv2.circle(frame, tuple(int(v) for v in point), radius, point_color, -1)
    if shoulder_px is not None and elbow_px is not None:
        cv2.line(frame, tuple(int(v) for v in shoulder_px), tuple(int(v) for v in elbow_px), (255, 220, 80), 2)
    if elbow_px is not None and wrist_px is not None:
        cv2.line(frame, tuple(int(v) for v in elbow_px), tuple(int(v) for v in wrist_px), (0, 170, 255), 3)


def extract_pose_points(args):
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    frame_stride = max(1, int(round(source_fps / args.fps)))
    target_frames = int(round(args.seconds * args.fps))
    raw_points = []
    raw_aux = []
    confidences = []
    overlay_frames = []
    frame_idx = 0
    kept = 0
    while kept < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride:
            frame_idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        wrist = elbow = shoulder = None
        confidence = 0.0
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            wrist_lm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            elbow_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            shoulder_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            wrist = np.array([wrist_lm.x, wrist_lm.y], dtype=float)
            elbow = np.array([elbow_lm.x, elbow_lm.y], dtype=float)
            shoulder = np.array([shoulder_lm.x, shoulder_lm.y], dtype=float)
            confidence = float(
                np.clip(
                    0.55 * getattr(wrist_lm, "visibility", 0.0)
                    + 0.25 * getattr(elbow_lm, "visibility", 0.0)
                    + 0.20 * getattr(shoulder_lm, "visibility", 0.0),
                    0.0,
                    1.0,
                )
            )
        raw_points.append(wrist if wrist is not None else [np.nan, np.nan])
        raw_aux.append(
            {
                "elbow": elbow.tolist() if elbow is not None else None,
                "shoulder": shoulder.tolist() if shoulder is not None else None,
            }
        )
        confidences.append(confidence)

        if args.overlay:
            display = frame.copy()
            wrist_px = (wrist * [width, height]).astype(int) if wrist is not None else None
            elbow_px = (elbow * [width, height]).astype(int) if elbow is not None else None
            shoulder_px = (shoulder * [width, height]).astype(int) if shoulder is not None else None
            draw_overlay(display, wrist_px, elbow_px, shoulder_px)
            cv2.putText(
                display,
                f"t={kept / args.fps:05.2f}s conf={confidence:.2f}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"t={kept / args.fps:05.2f}s conf={confidence:.2f}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            overlay_frames.append(display)
        kept += 1
        frame_idx += 1

    cap.release()
    pose.close()
    if len(raw_points) < 3:
        raise RuntimeError("Selected clip did not produce enough frames.")
    return np.asarray(raw_points, dtype=float), raw_aux, np.asarray(confidences, dtype=float), overlay_frames, (width, height)


def build_trajectory(args):
    config = load_config(args.retarget_config)
    robot_cfg = config.get("robot_world", {})
    robot_world_pos = np.asarray(robot_cfg.get("position", ROBOT_WORLD_POS), dtype=float)
    robot_world_yaw = float(robot_cfg.get("yaw", ROBOT_WORLD_YAW))
    task_cfg = config.get("task", {})
    cup_world = np.asarray(config.get("cup_world", CUP_WORLD), dtype=float)
    tcp_grasp = cup_world + np.asarray(task_cfg.get("tcp_grasp_offset_world", TCP_GRASP_OFFSET_WORLD), dtype=float)
    anchor_base = world_to_base(tcp_grasp, robot_world_pos, robot_world_yaw)

    raw_points, raw_aux, confidences, overlay_frames, frame_size = extract_pose_points(args)
    wrist_xy, valid = interpolate_missing(raw_points, confidences, args.min_confidence)
    wrist_xy = moving_average(wrist_xy, args.smooth_window)

    shoulder_xy = np.full_like(wrist_xy, np.nan)
    elbow_xy = np.full_like(wrist_xy, np.nan)
    for idx, aux in enumerate(raw_aux):
        if aux["shoulder"] is not None:
            shoulder_xy[idx] = aux["shoulder"]
        if aux["elbow"] is not None:
            elbow_xy[idx] = aux["elbow"]
    if np.isfinite(shoulder_xy[:, 0]).sum() >= 2:
        shoulder_xy, _ = interpolate_missing(shoulder_xy, np.ones(len(shoulder_xy)), 0.0)
    else:
        shoulder_xy[:] = np.nanmedian(wrist_xy, axis=0)

    wrist_center = np.nanmedian(wrist_xy, axis=0)
    x_span = max(0.12, float(np.nanpercentile(wrist_xy[:, 0], 90) - np.nanpercentile(wrist_xy[:, 0], 10)))
    y_span = max(0.10, float(np.nanpercentile(wrist_xy[:, 1], 90) - np.nanpercentile(wrist_xy[:, 1], 10)))
    extension = np.linalg.norm(wrist_xy - shoulder_xy, axis=1)
    ext_center = float(np.nanmedian(extension))
    ext_span = max(0.06, float(np.nanpercentile(extension, 90) - np.nanpercentile(extension, 10)))

    min_base = np.asarray(config.get("target_limits_base", {}).get("min", [-0.72, -0.68, 0.10]), dtype=object)
    max_base = np.asarray(config.get("target_limits_base", {}).get("max", [0.45, 0.25, 0.55]), dtype=object)
    safety = config.get("safety", {})
    min_tcp_z = float(safety.get("table", {}).get("min_tcp_z_base", args.min_tcp_z_base))

    samples = []
    base_points = []
    for idx, point in enumerate(wrist_xy):
        t = idx / args.fps
        horizontal = np.clip((point[0] - wrist_center[0]) / x_span, -1.0, 1.0)
        vertical = np.clip((wrist_center[1] - point[1]) / y_span, -1.0, 1.0)
        depth = np.clip((extension[idx] - ext_center) / ext_span, -1.0, 1.0)

        target_base = np.array(
            [
                anchor_base[0] + args.depth_scale * depth,
                anchor_base[1] + args.horizontal_scale * horizontal,
                args.base_z + args.vertical_scale * vertical,
            ],
            dtype=float,
        )
        target_base[2] = max(target_base[2], min_tcp_z)
        for axis in range(3):
            if min_base[axis] is not None:
                target_base[axis] = max(target_base[axis], float(min_base[axis]))
            if max_base[axis] is not None:
                target_base[axis] = min(target_base[axis], float(max_base[axis]))
        base_points.append(target_base)
        world = base_to_world(target_base, robot_world_pos, robot_world_yaw)
        samples.append(
            {
                "time": round(t, 4),
                "phase": phase_for_time(t, args.seconds),
                "right_hand_world": [round(float(v), 6) for v in world],
                "right_hand_base": [round(float(v), 6) for v in target_base],
                "right_hand_image": [round(float(v), 6) for v in point],
                "confidence": round(float(confidences[idx]), 4),
                "interpolated": not bool(valid[idx]),
            }
        )

    if args.overlay and overlay_frames:
        overlay_path = Path(args.overlay)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = overlay_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
        for frame in overlay_frames:
            writer.write(frame)
        writer.release()

    base_points = np.asarray(base_points)
    payload = {
        "schema": "smplx_d455_reconstructed_right_hand_trajectory/v1",
        "source": "single_view_taichi_video_mediapipe_pose",
        "video": str(args.video),
        "clip": {
            "start_seconds": args.start,
            "duration_seconds": args.seconds,
            "fps": args.fps,
            "frame_size": list(frame_size),
            "hand": "right",
        },
        "mapping": {
            "mode": "monocular_right_wrist_to_xarm_base_workspace",
            "anchor_base": [round(float(v), 6) for v in anchor_base],
            "base_z": args.base_z,
            "horizontal_scale": args.horizontal_scale,
            "vertical_scale": args.vertical_scale,
            "depth_scale": args.depth_scale,
            "min_tcp_z_base": min_tcp_z,
            "note": "This is monocular pose retargeting. It tracks the anatomical right wrist in RGB and maps normalized image motion into the xArm7 workspace; it is not RGB-D depth reconstruction.",
        },
        "quality": {
            "samples": len(samples),
            "confident_samples": int(valid.sum()),
            "confidence_mean": round(float(np.mean(confidences)), 4),
            "confidence_min": round(float(np.min(confidences)), 4),
            "base_min": [round(float(v), 6) for v in np.min(base_points, axis=0)],
            "base_max": [round(float(v), 6) for v in np.max(base_points, axis=0)],
        },
        "samples": samples,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Extract the middle right-hand trajectory from taichi.mp4 for xArm7 retargeting.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--retarget-config", type=Path)
    parser.add_argument("--start", type=float, default=78.0)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--base-z", type=float, default=0.28)
    parser.add_argument("--horizontal-scale", type=float, default=0.26)
    parser.add_argument("--vertical-scale", type=float, default=0.16)
    parser.add_argument("--depth-scale", type=float, default=0.16)
    parser.add_argument("--min-tcp-z-base", type=float, default=0.13)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()

    payload = build_trajectory(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"trajectory={args.out}")
    print(f"samples={payload['quality']['samples']} confident={payload['quality']['confident_samples']} mean_conf={payload['quality']['confidence_mean']}")
    print(f"base_min={payload['quality']['base_min']} base_max={payload['quality']['base_max']}")


if __name__ == "__main__":
    main()
