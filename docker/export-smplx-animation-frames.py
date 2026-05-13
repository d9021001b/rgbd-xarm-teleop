#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated SMPL-X animation frame\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f}\n")
        for face in faces:
            a, b, c = face + 1
            handle.write(f"f {a} {b} {c}\n")


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def load_hand_pose(path, name):
    if path is None:
        return None, None
    data = np.load(path, allow_pickle=True)
    return data["hand_poses"].item()[name]


def build_body_poses(frame_count, fps):
    ready = np.zeros(63, dtype=np.float32)
    hover = np.zeros(63, dtype=np.float32)
    grasp = np.zeros(63, dtype=np.float32)
    lift = np.zeros(63, dtype=np.float32)

    for pose in (ready, hover, grasp, lift):
        pose[15 * 3 : 15 * 3 + 3] = [0.0, 0.0, -1.40]

    ready[16 * 3 : 16 * 3 + 3] = [-0.40, -0.60, 0.20]
    ready[18 * 3 : 18 * 3 + 3] = [0.0, 0.0, -0.30]
    ready[20 * 3 : 20 * 3 + 3] = [0.0, 0.0, 0.16]

    hover[16 * 3 : 16 * 3 + 3] = [0.40, 0.40, 0.20]
    hover[18 * 3 : 18 * 3 + 3] = [0.0, 0.0, -0.30]
    hover[20 * 3 : 20 * 3 + 3] = [0.0, 0.0, 0.16]

    grasp[16 * 3 : 16 * 3 + 3] = [0.40, 0.40, 0.30]
    grasp[18 * 3 : 18 * 3 + 3] = [0.0, 0.0, -0.30]
    grasp[20 * 3 : 20 * 3 + 3] = [0.0, 0.0, 0.16]

    lift[16 * 3 : 16 * 3 + 3] = [0.40, 0.40, 0.0]
    lift[18 * 3 : 18 * 3 + 3] = [0.0, 0.0, -0.40]
    lift[20 * 3 : 20 * 3 + 3] = [0.0, 0.0, 0.18]

    poses = []
    phases = []
    for frame in range(frame_count):
        t = frame / fps
        if t < 1.0:
            pose = ready
            phase = "ready"
        elif t < 4.8:
            alpha = smoothstep((t - 1.0) / 3.8)
            pose = ready * (1.0 - alpha) + hover * alpha
            phase = "over-table-approach"
        elif t < 6.4:
            alpha = smoothstep((t - 4.8) / 1.6)
            pose = hover * (1.0 - alpha) + grasp * alpha
            phase = "descend"
        elif t < 7.4:
            pose = grasp
            phase = "grasp"
        else:
            alpha = smoothstep((t - 7.4) / 2.6)
            pose = grasp * (1.0 - alpha) + lift * alpha
            phase = "lift"
        poses.append(pose.copy())
        phases.append(phase)
    return np.asarray(poses, dtype=np.float32), phases


def main():
    parser = argparse.ArgumentParser(description="Export a SMPL-X cup-grasp motion as Gazebo OBJ frame sequence.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hand-poses")
    parser.add_argument("--out-model-dir", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    args = parser.parse_args()

    try:
        import torch
        import smplx
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Activate .venv-smplx with smplx, torch, numpy.") from exc

    frame_count = int(round(args.seconds * args.fps)) + 1
    body_poses, phases = build_body_poses(frame_count, args.fps)
    left_hand, right_hand = load_hand_pose(args.hand_poses, "relaxed")

    model = smplx.create(
        args.model_dir,
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        batch_size=frame_count,
    )
    tensor_params = {"body_pose": torch.as_tensor(body_poses, dtype=torch.float32)}
    if left_hand is not None:
        tensor_params["left_hand_pose"] = torch.as_tensor(
            np.tile(left_hand.reshape(1, -1), (frame_count, 1)), dtype=torch.float32
        )
    if right_hand is not None:
        tensor_params["right_hand_pose"] = torch.as_tensor(
            np.tile(right_hand.reshape(1, -1), (frame_count, 1)), dtype=torch.float32
        )

    with torch.no_grad():
        output = model(return_verts=True, **tensor_params)
    vertices_by_frame = output.vertices.detach().cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int64)

    out_model_dir = Path(args.out_model_dir).expanduser().resolve()
    mesh_dir = out_model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old in mesh_dir.glob("frame_*.obj"):
        old.unlink()

    frame_records = []
    for frame in range(frame_count):
        obj_name = f"frame_{frame:04d}.obj"
        write_obj(mesh_dir / obj_name, vertices_by_frame[frame], faces)
        frame_records.append(
            {
                "frame": frame,
                "time": round(frame / args.fps, 4),
                "mesh": f"meshes/{obj_name}",
                "phase": phases[min(frame, len(phases) - 1)],
            }
        )

    (out_model_dir / "model.config").write_text(
        """<?xml version="1.0"?>
<model>
  <name>smplx_operator_animation</name>
  <version>1.0</version>
  <sdf version="1.10">model.sdf</sdf>
  <author>
    <name>Local xArm7 Gazebo setup</name>
  </author>
  <description>SMPL-X OBJ frame sequence for Gazebo animation.</description>
</model>
""",
        encoding="utf-8",
    )
    (out_model_dir / "model.sdf").write_text(
        """<?xml version="1.0"?>
<sdf version="1.10">
  <model name="smplx_operator_animation">
    <static>true</static>
    <link name="body">
      <visual name="frame_visual">
        <pose>0 0 1.26 1.5708 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://smplx_operator_animation/meshes/frame_0000.obj</uri>
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
</sdf>
""",
        encoding="utf-8",
    )
    manifest = {
        "model": "smplx_operator_animation",
        "seconds": args.seconds,
        "fps": args.fps,
        "frames": frame_count,
        "pose": {
            "x": -0.283,
            "y": -1.127,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 3.55,
        },
        "hidden_pose": {"x": 0.0, "y": 0.0, "z": -30.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "frames_detail": frame_records,
    }
    (out_model_dir / "animation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported_smplx_animation_frames={frame_count} output={out_model_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
