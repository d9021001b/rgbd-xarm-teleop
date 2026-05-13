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


def image_to_array(msg):
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8UC1": 1,
        "16UC1": 1,
        "32FC1": 1,
    }
    dtype_by_encoding = {
        "rgb8": np.uint8,
        "bgr8": np.uint8,
        "rgba8": np.uint8,
        "bgra8": np.uint8,
        "mono8": np.uint8,
        "8UC1": np.uint8,
        "16UC1": np.uint16,
        "32FC1": np.float32,
    }
    channels = channels_by_encoding.get(msg.encoding, 3)
    dtype = dtype_by_encoding.get(msg.encoding, np.uint8)
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        return arr.reshape((msg.height, msg.width))
    return arr.reshape((msg.height, msg.width, channels))


class D455VideoRecorder(Node):
    def __init__(self, seconds, out_dir):
        super().__init__("d455_video_recorder")
        self.seconds = seconds
        self.started_at = time.monotonic()
        self.deadline = time.monotonic() + seconds
        self.out_dir = Path(out_dir)
        self.rgb_frames_buffer = []
        self.depth_frames_buffer = []
        self.rgb_frames = 0
        self.depth_frames = 0
        self.rgb_topic = "/tripod_d455/depth/image"
        self.depth_topic = "/tripod_d455/depth/depth_image"
        self.create_subscription(Image, self.rgb_topic, self.on_rgb, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)

    def _write_video(self, filename, frames, elapsed):
        if not frames:
            return 0.0
        height, width = frames[0].shape[:2]
        path = str(self.out_dir / filename)
        fps = round(min(60.0, max(1.0, len(frames) / max(0.1, elapsed))), 2)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        return fps

    def on_rgb(self, msg):
        frame = image_to_array(msg)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            frame = frame[:, :, :3]
        self.rgb_frames_buffer.append(frame.copy())
        self.rgb_frames += 1

    def on_depth(self, msg):
        depth = image_to_array(msg)
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        clipped = np.clip(depth_m, 0.1, 8.0)
        normalized = ((clipped - 0.1) / (8.0 - 0.1) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        self.depth_frames_buffer.append(colored.copy())
        self.depth_frames += 1

    def done(self):
        return time.monotonic() >= self.deadline

    def close(self):
        elapsed = max(0.1, time.monotonic() - self.started_at)
        rgb_fps = self._write_video("d455_rgb.mp4", self.rgb_frames_buffer, elapsed)
        depth_fps = self._write_video("d455_depth_colormap.mp4", self.depth_frames_buffer, elapsed)
        manifest = {
            "rgb_topic": self.rgb_topic,
            "depth_topic": self.depth_topic,
            "rgb_video": "d455_rgb.mp4",
            "depth_video": "d455_depth_colormap.mp4",
            "rgb_frames": self.rgb_frames,
            "depth_frames": self.depth_frames,
            "elapsed_seconds": elapsed,
            "rgb_video_fps": rgb_fps,
            "depth_video_fps": depth_fps,
            "note": "PointCloud2 is recorded in the ROS bag directory, not converted to MP4.",
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = D455VideoRecorder(args.seconds, out_dir)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
