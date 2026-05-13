#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


LANDMARKS = {
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
}


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    out = np.zeros_like(values)
    for landmark in range(values.shape[1]):
        for axis in range(3):
            out[:, landmark, axis] = np.convolve(padded[:, landmark, axis], kernel, mode="valid")
    return out


def interpolate_missing(points, confidences, min_confidence):
    points = np.asarray(points, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    valid = np.isfinite(points[:, 0, 0]) & (confidences >= min_confidence)
    if valid.sum() < 2:
        raise RuntimeError("Not enough confident MediaPipe 3D pose frames in selected clip.")
    xs = np.arange(points.shape[0])
    filled = points.copy()
    for landmark in range(points.shape[1]):
        for axis in range(3):
            filled[:, landmark, axis] = np.interp(xs, xs[valid], points[valid, landmark, axis])
    return filled, valid


def resolve_roi(width, height, roi_normalized, roi_pixels):
    if roi_normalized and roi_pixels:
        raise ValueError("Use only one of --roi-normalized or --roi-pixels.")
    if roi_pixels:
        x, y, w, h = roi_pixels
    elif roi_normalized:
        x, y, w, h = roi_normalized
        x, y, w, h = x * width, y * height, w * width, h * height
    else:
        x, y, w, h = 0, 0, width, height
    x0 = int(round(max(0, min(width - 2, x))))
    y0 = int(round(max(0, min(height - 2, y))))
    x1 = int(round(max(x0 + 2, min(width, x + w))))
    y1 = int(round(max(y0 + 2, min(height, y + h))))
    return x0, y0, x1, y1


def draw_overlay(frame, image_points, confidence, t, roi_box=None):
    if roi_box:
        x0, y0, x1, y1 = roi_box
        if (x0, y0, x1, y1) != (0, 0, frame.shape[1], frame.shape[0]):
            cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 210, 60), 2, cv2.LINE_AA)
    color = (0, 150, 255)
    for a, b in (
        ("left_shoulder", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ):
        if image_points.get(a) is not None and image_points.get(b) is not None:
            cv2.line(frame, image_points[a], image_points[b], color, 3, cv2.LINE_AA)
    for name, point in image_points.items():
        if point is None:
            continue
        radius = 7 if name == "right_wrist" else 5
        point_color = (0, 80, 255) if name == "right_wrist" else (0, 220, 255)
        cv2.circle(frame, point, radius, point_color, -1, cv2.LINE_AA)
    label = f"MediaPipe Pose 3D | t={t:05.2f}s conf={confidence:.2f}"
    cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Pose 3D landmarks from a selected person in taichi.mp4.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--start", type=float, default=78.0)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--roi-normalized",
        nargs=4,
        type=float,
        metavar=("X", "Y", "W", "H"),
        help="Crop MediaPipe input to a normalized full-frame ROI before pose detection.",
    )
    parser.add_argument(
        "--roi-pixels",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Crop MediaPipe input to a pixel ROI before pose detection.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_box = resolve_roi(width, height, args.roi_normalized, args.roi_pixels)
    roi_x0, roi_y0, roi_x1, roi_y1 = roi_box
    roi_width = roi_x1 - roi_x0
    roi_height = roi_y1 - roi_y0
    frame_stride = max(1, int(round(source_fps / args.fps)))
    target_frames = int(round(args.seconds * args.fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    landmark_names = list(LANDMARKS.keys())
    world_points = []
    image_points = []
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
        input_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        world_row = np.full((len(landmark_names), 3), np.nan, dtype=float)
        image_row = {}
        confidence_values = []
        if result.pose_landmarks and result.pose_world_landmarks:
            for idx, name in enumerate(landmark_names):
                landmark_idx = LANDMARKS[name].value
                world_lm = result.pose_world_landmarks.landmark[landmark_idx]
                image_lm = result.pose_landmarks.landmark[landmark_idx]
                world_row[idx] = [world_lm.x, world_lm.y, world_lm.z]
                image_row[name] = (
                    int(roi_x0 + image_lm.x * roi_width),
                    int(roi_y0 + image_lm.y * roi_height),
                )
                confidence_values.append(float(getattr(image_lm, "visibility", 0.0)))
        for name in landmark_names:
            image_row.setdefault(name, None)
        confidence = float(np.clip(np.mean(confidence_values), 0.0, 1.0)) if confidence_values else 0.0
        world_points.append(world_row)
        image_points.append(image_row)
        confidences.append(confidence)

        if args.overlay:
            display = frame.copy()
            draw_overlay(display, image_row, confidence, kept / args.fps, roi_box=roi_box)
            overlay_frames.append(display)
        kept += 1
        frame_idx += 1

    cap.release()
    pose.close()
    if len(world_points) < 3:
        raise RuntimeError("Selected clip did not produce enough frames.")

    world_points = np.asarray(world_points, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    filled_points, valid = interpolate_missing(world_points, confidences, args.min_confidence)
    smoothed_points = moving_average(filled_points, args.smooth_window)

    samples = []
    for frame, points in enumerate(smoothed_points):
        landmarks = {}
        for idx, name in enumerate(landmark_names):
            landmarks[name] = [round(float(v), 7) for v in points[idx]]
        samples.append(
            {
                "frame": frame,
                "time": round(frame / args.fps, 4),
                "confidence": round(float(confidences[frame]), 4),
                "interpolated": not bool(valid[frame]),
                "landmarks_world": landmarks,
            }
        )

    payload = {
        "schema": "mediapipe_pose3d_right_arm/v1",
        "source": "mediapipe_pose_world_landmarks",
        "video": str(args.video),
        "clip": {
            "start_seconds": args.start,
            "duration_seconds": args.seconds,
            "fps": args.fps,
            "frame_size": [width, height],
            "roi_pixels": [roi_x0, roi_y0, roi_width, roi_height],
            "roi_normalized": [
                round(float(roi_x0 / width), 6),
                round(float(roi_y0 / height), 6),
                round(float(roi_width / width), 6),
                round(float(roi_height / height), 6),
            ],
        },
        "landmark_names": landmark_names,
        "quality": {
            "samples": len(samples),
            "confident_samples": int(valid.sum()),
            "confidence_mean": round(float(np.mean(confidences)), 4),
            "confidence_min": round(float(np.min(confidences)), 4),
        },
        "note": "MediaPipe pose_world_landmarks are metric-like body-centric coordinates, not camera-calibrated D455 depth.",
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.overlay and overlay_frames:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.overlay), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
        for frame in overlay_frames:
            writer.write(frame)
        writer.release()
    print(f"skeleton={args.out}")
    print(f"samples={payload['quality']['samples']} confident={payload['quality']['confident_samples']} mean_conf={payload['quality']['confidence_mean']}")


if __name__ == "__main__":
    main()
