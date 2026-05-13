#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def decimate_mesh(vertices, faces, face_stride):
    face_stride = max(1, int(face_stride))
    if face_stride == 1:
        return vertices, faces
    faces = faces[::face_stride]
    used = np.unique(faces.reshape(-1))
    remap = {int(old): idx for idx, old in enumerate(used)}
    compact_faces = np.asarray([[remap[int(index)] for index in face] for face in faces], dtype=np.int64)
    return vertices[used], compact_faces


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated fitted SMPL-X animation frame\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f}\n")
        for face in faces:
            a, b, c = face + 1
            handle.write(f"f {a} {b} {c}\n")


def load_hand_pose(path, name):
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    pose = data["hand_poses"].item()[name]
    if isinstance(pose, tuple):
        pose = pose[0]
    return np.asarray(pose, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Export fitted SMPL-X body-pose frames as Gazebo OBJ sequence.")
    parser.add_argument("--body-pose-npy", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--hand-poses", type=Path)
    parser.add_argument("--out-model-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    args = parser.parse_args()

    try:
        import smplx
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Activate .venv-smplx with smplx, torch, numpy.") from exc

    body_pose = np.load(args.body_pose_npy).astype(np.float32)
    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise SystemExit(f"Expected body_pose shape (frames, 63), got {body_pose.shape}")
    frame_count = int(body_pose.shape[0])

    left_hand = load_hand_pose(args.hand_poses, "relaxed")
    right_hand = load_hand_pose(args.hand_poses, "relaxed")
    model = smplx.create(
        str(args.model_dir),
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        batch_size=frame_count,
    )
    tensor_params = {"body_pose": torch.as_tensor(body_pose, dtype=torch.float32)}
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
    out_model_dir = args.out_model_dir.expanduser().resolve()
    mesh_dir = out_model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for old in mesh_dir.glob("frame_*.obj"):
        old.unlink()

    duration = (frame_count - 1) / max(1e-6, args.fps)
    frame_records = []
    for frame in range(frame_count):
        obj_name = f"frame_{frame:04d}.obj"
        vertices, compact_faces = decimate_mesh(vertices_by_frame[frame], faces, args.face_stride)
        write_obj(mesh_dir / obj_name, vertices, compact_faces)
        frame_records.append(
            {
                "frame": frame,
                "time": round(frame / args.fps, 4),
                "mesh": f"meshes/{obj_name}",
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
  <description>Fitted SMPL-X OBJ frame sequence for Gazebo animation.</description>
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
    pose = args.pose or [-0.283, -1.127, 0.0, 0.0, 0.0, 3.55]
    manifest = {
        "model": "smplx_operator_animation",
        "source": "fitted_smplx_pose_from_mediapipe3d",
        "body_pose_npy": str(args.body_pose_npy),
        "seconds": round(float(duration), 4),
        "fps": args.fps,
        "face_stride": args.face_stride,
        "frames": frame_count,
        "pose": {
            "x": pose[0],
            "y": pose[1],
            "z": pose[2],
            "roll": pose[3],
            "pitch": pose[4],
            "yaw": pose[5],
        },
        "hidden_pose": {"x": 0.0, "y": 0.0, "z": -30.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "frames_detail": frame_records,
    }
    (out_model_dir / "animation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported_fitted_smplx_animation_frames={frame_count} output={out_model_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
