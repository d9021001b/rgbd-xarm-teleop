#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def image_to_bgr(msg):
    dtype = np.uint8
    channels = 3
    if msg.encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif msg.encoding in {"mono8", "8UC1"}:
        channels = 1
    elif msg.encoding == "16UC1":
        dtype = np.uint16
        channels = 1

    frame = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        frame = frame.reshape((msg.height, msg.width))
        if dtype == np.uint16:
            frame = np.clip(frame.astype(np.float32) / 4000.0 * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    frame = frame.reshape((msg.height, msg.width, channels))
    if msg.encoding == "rgb8":
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
    if msg.encoding == "rgba8":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if msg.encoding == "bgra8":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame[:, :, :3]


class CloseupRecorder(Node):
    def __init__(self, topic, seconds, out_dir, filename, fps):
        super().__init__("retarget_closeup_recorder")
        self.topic = topic
        self.seconds = seconds
        self.out_dir = Path(out_dir)
        self.filename = filename
        self.fps = float(fps)
        self.created_at = time.monotonic()
        self.started_at = None
        self.frame_count = 0
        self.writer = None
        self.frame_size = None
        self.last_frame = None
        self.create_subscription(Image, topic, self.on_image, 10)

    def on_image(self, msg):
        now = time.monotonic()
        if self.started_at is None:
            self.started_at = now
        frame = image_to_bgr(msg)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
            output = self.out_dir / self.filename
            self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.frame_size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {output}")
        target_index = int((now - self.started_at) * self.fps)
        while self.frame_count < target_index and self.last_frame is not None:
            self.writer.write(self.last_frame)
            self.frame_count += 1
        self.writer.write(frame)
        self.frame_count += 1
        self.last_frame = frame

    def done(self):
        if self.started_at is None:
            return time.monotonic() - self.created_at > 10.0
        return time.monotonic() - self.started_at >= self.seconds

    def close(self):
        started_at = self.started_at if self.started_at is not None else self.created_at
        elapsed = max(0.1, time.monotonic() - started_at)
        if self.writer is not None:
            target_total = int(math.ceil(elapsed * self.fps))
            while self.frame_count < target_total and self.last_frame is not None:
                self.writer.write(self.last_frame)
                self.frame_count += 1
            self.writer.release()
        manifest = {
            "topic": self.topic,
            "video": self.filename,
            "frames": self.frame_count,
            "elapsed_seconds": elapsed,
            "fps": self.fps,
            "created_monotonic_time": self.created_at,
            "started_monotonic_time": self.started_at,
            "closed_monotonic_time": time.monotonic(),
            "note": "Native Gazebo close-up camera recording with wall-clock frame padding for timing preservation.",
        }
        (self.out_dir / "retarget_closeup_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/retarget_closeup/image")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--filename", default="gazebo_xarm_smplx_retarget.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = CloseupRecorder(args.topic, args.seconds, out_dir, args.filename, args.fps)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
