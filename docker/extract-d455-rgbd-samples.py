#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import Image


RGB_TOPIC = "/tripod_d455/depth/image"
DEPTH_TOPIC = "/tripod_d455/depth/depth_image"


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


def rgb_to_array(msg):
    frame = image_to_array(msg)
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if msg.encoding == "bgr8":
        return frame[:, :, [2, 1, 0]]
    if msg.encoding == "bgra8":
        return frame[:, :, [2, 1, 0]]
    return frame[:, :, :3]


def depth_to_meters(msg):
    depth = image_to_array(msg)
    if msg.encoding == "16UC1":
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def open_reader(bag_dir):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def main():
    parser = argparse.ArgumentParser(description="Extract paired D455 RGB and raw depth frames from a ROS 2 MCAP bag.")
    parser.add_argument("--bag", required=True, help="Path to d455_rosbag directory.")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--fps", type=float, default=3.0, help="Sample FPS for fitting.")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--max-skew", type=float, default=1.0, help="Maximum RGB/depth timestamp skew to accept.")
    parser.add_argument("--rgb-topic", default=RGB_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    args = parser.parse_args()

    rclpy.init()
    try:
        reader = open_reader(Path(args.bag))
        rgb_frames = []
        rgb_times = []
        depth_frames = []
        depth_times = []
        start_ns = None
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic not in (args.rgb_topic, args.depth_topic):
                continue
            if start_ns is None:
                start_ns = timestamp
            t = (timestamp - start_ns) * 1e-9
            if t > args.max_seconds:
                break
            if topic == args.rgb_topic:
                msg = deserialize_message(data, Image)
                rgb_frames.append(rgb_to_array(msg))
                rgb_times.append(t)
            elif topic == args.depth_topic:
                msg = deserialize_message(data, Image)
                depth_frames.append(depth_to_meters(msg))
                depth_times.append(t)
    finally:
        rclpy.shutdown()

    if not rgb_frames or not depth_frames:
        raise RuntimeError("No RGB/depth frames found in bag.")

    rgb_times = np.asarray(rgb_times, dtype=np.float64)
    depth_times = np.asarray(depth_times, dtype=np.float64)
    sample_times = np.arange(0.0, min(args.max_seconds, float(rgb_times[-1])), 1.0 / args.fps)
    paired_rgb = []
    paired_depth = []
    paired_times = []
    for t in sample_times:
        rgb_idx = int(np.argmin(np.abs(rgb_times - t)))
        depth_idx = int(np.argmin(np.abs(depth_times - t)))
        if abs(float(rgb_times[rgb_idx] - t)) > args.max_skew or abs(float(depth_times[depth_idx] - t)) > args.max_skew:
            continue
        paired_rgb.append(rgb_frames[rgb_idx])
        paired_depth.append(depth_frames[depth_idx])
        paired_times.append(t)

    if not paired_rgb:
        raise RuntimeError("No paired frames could be sampled.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rgb=np.asarray(paired_rgb, dtype=np.uint8),
        depth_m=np.asarray(paired_depth, dtype=np.float32),
        times=np.asarray(paired_times, dtype=np.float32),
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
    )
    print(f"rgbd_samples={len(paired_rgb)} output={out}")


if __name__ == "__main__":
    raise SystemExit(main())
