#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import time
from pathlib import Path


GZ_BIN = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
VISIBLE_POSE = (-0.283, -1.127, 0.0, 0.0, 0.0, 3.55)
HIDDEN_Z = -30.0


def gz_service(service, reqtype, reptype, req, timeout=5000):
    result = subprocess.run(
        [
            GZ_BIN,
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            str(timeout),
            "--req",
            req,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def pose_req(name, pose):
    return pose_block(name, pose)


def pose_block(name, pose):
    x, y, z, roll, pitch, yaw = pose
    return (
        f'name: "{name}" '
        f"position: {{x: {x:.6f} y: {y:.6f} z: {z:.6f}}} "
        f"orientation: {{x: 0 y: 0 z: {math.sin(yaw * 0.5):.8f} "
        f"w: {math.cos(yaw * 0.5):.8f}}}"
    )


def set_pose(name, pose):
    return gz_service(
        "/world/default/set_pose",
        "gz.msgs.Pose",
        "gz.msgs.Boolean",
        pose_req(name, pose),
        timeout=3000,
    )


def set_pose_vector(named_poses):
    if not named_poses:
        return 0, "data: true", ""
    req = " ".join(f"pose: {{{pose_block(name, pose)}}}" for name, pose in named_poses)
    return gz_service(
        "/world/default/set_pose_vector",
        "gz.msgs.Pose_V",
        "gz.msgs.Boolean",
        req,
        timeout=2000,
    )


def remove_model(name):
    req = f'name: "{name}" type: MODEL'
    return gz_service(
        "/world/default/remove/blocking",
        "gz.msgs.Entity",
        "gz.msgs.Boolean",
        req,
        timeout=5000,
    )


def create_model(name, mesh_uri):
    sdf = f"""<sdf version='1.10'>
  <model name='{name}'>
    <static>true</static>
    <pose>0 0 {HIDDEN_Z:.3f} 0 0 0</pose>
    <link name='body'>
      <visual name='smplx_frame_visual'>
        <pose>0 0 1.26 1.5708 0 0</pose>
        <geometry>
          <mesh>
            <uri>{mesh_uri}</uri>
          </mesh>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.08 0.08 0.08 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    req = "sdf: " + json.dumps(sdf)
    return gz_service(
        "/world/default/create/blocking",
        "gz.msgs.EntityFactory",
        "gz.msgs.Boolean",
        req,
        timeout=60000,
    )


def ensure_models(frame_count, start_frame=0):
    for idx in range(int(start_frame), frame_count):
        name = f"smplx_anim_frame_{idx:04d}"
        remove_model(name)
        uri = f"model://smplx_operator_animation/meshes/frame_{idx:04d}.obj"
        code, out, err = create_model(name, uri)
        if code != 0 or "data: true" not in out:
            raise RuntimeError(f"Could not create {name}: code={code} stdout={out} stderr={err}")


def hide_all(frame_count):
    batch = []
    for idx in range(frame_count):
        batch.append((f"smplx_anim_frame_{idx:04d}", (0.0, 0.0, HIDDEN_Z - idx * 0.02, 0.0, 0.0, 0.0)))
        if len(batch) >= 12:
            set_pose_vector(batch)
            batch = []
    set_pose_vector(batch)


def main():
    parser = argparse.ArgumentParser(description="Animate Gazebo SMPL-X by switching visible OBJ frame models.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--visible-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    parser.add_argument("--ensure-models", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--hide-static", action="store_true")
    parser.add_argument("--skip-hide-all", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frame_count = int(manifest["frames"])
    manifest_pose = manifest.get("pose", {})
    visible_pose = tuple(
        args.visible_pose
        or (
            manifest_pose.get("x", VISIBLE_POSE[0]),
            manifest_pose.get("y", VISIBLE_POSE[1]),
            manifest_pose.get("z", VISIBLE_POSE[2]),
            manifest_pose.get("roll", VISIBLE_POSE[3]),
            manifest_pose.get("pitch", VISIBLE_POSE[4]),
            manifest_pose.get("yaw", VISIBLE_POSE[5]),
        )
    )
    if args.hide_static:
        set_pose("smplx_operator_visual", (0.0, 0.0, HIDDEN_Z - 5.0, 0.0, 0.0, 0.0))
    if args.ensure_models:
        ensure_models(frame_count, start_frame=args.start_frame)

    if not args.skip_hide_all:
        hide_all(frame_count)
    if args.prepare_only:
        print(f"smplx_animation_prepared frames={frame_count}")
        return 0

    if args.start_delay > 0.0:
        time.sleep(args.start_delay)
    interval = 1.0 / args.fps
    start = time.monotonic()
    previous = None
    updates = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed > args.seconds:
            break
        source_elapsed = elapsed / max(1e-6, args.time_scale)
        idx = min(frame_count - 1, int(round(source_elapsed * args.fps)))
        if idx != previous:
            batch = []
            if previous is not None:
                batch.append((f"smplx_anim_frame_{previous:04d}", (0.0, 0.0, HIDDEN_Z - previous * 0.02, 0.0, 0.0, 0.0)))
            batch.append((f"smplx_anim_frame_{idx:04d}", visible_pose))
            code, out, err = set_pose_vector(batch)
            if code != 0 or "data: true" not in out:
                print(f"set_pose_vector_failed frame={idx} code={code} stdout={out} stderr={err}")
            previous = idx
            updates += 1
        time.sleep(max(0.01, interval * 0.35))
    if previous is not None:
        set_pose(f"smplx_anim_frame_{previous:04d}", visible_pose)
    print(f"smplx_animation_updates={updates} frames={frame_count} seconds={args.seconds} fps={args.fps}")


if __name__ == "__main__":
    raise SystemExit(main())
