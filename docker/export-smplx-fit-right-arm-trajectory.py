#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


JOINTS = {
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
}

SMPLX_VISUAL_Z = 1.30


def smplx_to_world(points):
    points = np.asarray(points, dtype=np.float32)
    return np.column_stack([points[:, 0], -points[:, 2], points[:, 1] + SMPLX_VISUAL_Z])


def phase_for_time(t, duration):
    value = t / max(1e-6, duration)
    if value < 0.18:
        return "taichi-prepare"
    if value < 0.45:
        return "taichi-reach"
    if value < 0.72:
        return "taichi-sweep"
    return "taichi-return"


def smplx_forward(model, torch, body_pose, return_verts=False):
    batch = int(body_pose.shape[0])
    dtype = body_pose.dtype
    device = body_pose.device
    params = {
        "body_pose": body_pose,
        "global_orient": torch.zeros((batch, 3), dtype=dtype, device=device),
        "transl": torch.zeros((batch, 3), dtype=dtype, device=device),
        "betas": torch.zeros((batch, getattr(model, "num_betas", 10)), dtype=dtype, device=device),
        "expression": torch.zeros((batch, getattr(model, "num_expression_coeffs", 10)), dtype=dtype, device=device),
        "jaw_pose": torch.zeros((batch, 3), dtype=dtype, device=device),
        "leye_pose": torch.zeros((batch, 3), dtype=dtype, device=device),
        "reye_pose": torch.zeros((batch, 3), dtype=dtype, device=device),
        "left_hand_pose": torch.zeros((batch, 45), dtype=dtype, device=device),
        "right_hand_pose": torch.zeros((batch, 45), dtype=dtype, device=device),
    }
    return model(return_verts=return_verts, **params)


def rounded_vector(values):
    return [round(float(value), 6) for value in values]


def vector_range(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": rounded_vector(values.min(axis=0)),
        "max": rounded_vector(values.max(axis=0)),
        "range": rounded_vector(values.max(axis=0) - values.min(axis=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Export fitted SMPL-X right-arm joints as the xArm7 retarget trajectory contract.")
    parser.add_argument("--body-pose-npy", required=True, type=Path)
    parser.add_argument("--skeleton-json", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    args = parser.parse_args()

    try:
        import smplx
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Activate .venv-smplx with smplx and torch.") from exc

    body_pose = np.load(args.body_pose_npy).astype(np.float32)
    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise SystemExit(f"Expected body_pose shape (frames, 63), got {body_pose.shape}")
    frame_count = int(body_pose.shape[0])
    skeleton = None
    if args.skeleton_json and args.skeleton_json.exists():
        skeleton = json.loads(args.skeleton_json.read_text(encoding="utf-8"))
        samples = skeleton.get("samples", [])
        if samples and len(samples) != frame_count:
            raise SystemExit(f"Skeleton samples ({len(samples)}) do not match body pose frames ({frame_count})")

    model = smplx.create(
        str(args.model_dir),
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        batch_size=frame_count,
    )
    with torch.no_grad():
        output = smplx_forward(model, torch, torch.as_tensor(body_pose, dtype=torch.float32), return_verts=False)
    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    duration = (frame_count - 1) / max(1e-6, args.fps)

    samples = []
    wrists = []
    elbows = []
    shoulders = []
    for frame in range(frame_count):
        t = frame / args.fps
        world_joints = smplx_to_world(joints[frame])
        shoulder = world_joints[JOINTS["right_shoulder"]]
        elbow = world_joints[JOINTS["right_elbow"]]
        wrist = world_joints[JOINTS["right_wrist"]]
        confidence = 1.0
        if skeleton is not None:
            confidence = float(skeleton["samples"][frame].get("confidence", 1.0))
        shoulder_to_elbow = elbow - shoulder
        elbow_to_wrist = wrist - elbow
        shoulder_to_wrist = wrist - shoulder
        samples.append(
            {
                "frame": frame,
                "time": round(float(t), 4),
                "phase": phase_for_time(t, duration),
                "right_shoulder_world": rounded_vector(shoulder),
                "right_elbow_world": rounded_vector(elbow),
                "right_wrist_world": rounded_vector(wrist),
                "right_hand_world": rounded_vector(wrist),
                "right_upper_arm_vector_world": rounded_vector(shoulder_to_elbow),
                "right_forearm_vector_world": rounded_vector(elbow_to_wrist),
                "right_shoulder_to_wrist_world": rounded_vector(shoulder_to_wrist),
                "confidence": round(float(confidence), 4),
            }
        )
        shoulders.append(shoulder)
        elbows.append(elbow)
        wrists.append(wrist)

    payload = {
        "schema": "smplx_fit_right_arm_trajectory/v1",
        "source": "fitted_smplx_pose_from_mediapipe3d",
        "body_pose_npy": str(args.body_pose_npy),
        "skeleton_json": str(args.skeleton_json) if args.skeleton_json else None,
        "fps": args.fps,
        "frames": frame_count,
        "duration_seconds": round(float(duration), 4),
        "coordinate_frame": {
            "world": "gazebo_default_world_compatible_visual_frame",
            "source": "SMPL-X native Y-up joints",
            "transform": "world=[smplx_x, -smplx_z, smplx_y + 1.30]",
        },
        "joint_indices": JOINTS,
        "quality": {
            "confidence_mean": round(float(np.mean([sample["confidence"] for sample in samples])), 4),
            "confidence_min": round(float(np.min([sample["confidence"] for sample in samples])), 4),
            "right_shoulder_world": vector_range(np.asarray(shoulders)),
            "right_elbow_world": vector_range(np.asarray(elbows)),
            "right_wrist_world": vector_range(np.asarray(wrists)),
        },
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
