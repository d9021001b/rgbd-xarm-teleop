#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import smplx
import torch


WIDTH = 848
HEIGHT = 480
HFOV = 2.05

TRIPOD_POS = np.array([-1.20, -3.10, 0.0], dtype=np.float32)
TRIPOD_RPY = np.array([0.0, 0.0, 1.12], dtype=np.float32)
SENSOR_POS = np.array([0.08, 0.0, 1.44], dtype=np.float32)
SENSOR_RPY = np.array([0.0, 0.12, 0.0], dtype=np.float32)

SMPLX_MODEL_POS = np.array([-0.283, -1.34, 0.0], dtype=np.float32)
SMPLX_MODEL_YAW = 3.55
SMPLX_VISUAL_Z = 1.26

RIGHT_HAND_VERTEX_RANGE = np.arange(7331, 8129, dtype=np.int64)
POSE_FIT_JOINTS = (16, 18, 20)


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def rpy_matrix(rpy):
    return rz(float(rpy[2])) @ ry(float(rpy[1])) @ rx(float(rpy[0]))


def camera_pose():
    tripod_r = rpy_matrix(TRIPOD_RPY)
    sensor_r = tripod_r @ rpy_matrix(SENSOR_RPY)
    sensor_pos = TRIPOD_POS + tripod_r @ SENSOR_POS
    return sensor_pos.astype(np.float32), sensor_r.astype(np.float32)


def backproject_mask(rgb, depth_m, max_points, rng):
    # The SMPL-X operator material is intentionally white in Gazebo. Use RGB to
    # isolate human pixels, then use raw D455 depth for metric 3D points.
    white = (rgb[:, :, 0] > 232) & (rgb[:, :, 1] > 232) & (rgb[:, :, 2] > 232)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.15) & (depth_m < 6.0)
    mask = white & valid_depth
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), mask
    if len(xs) > max_points:
        chosen = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[chosen]
        ys = ys[chosen]
    depth = depth_m[ys, xs].astype(np.float32)
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    local = np.column_stack(
        [
            depth,
            -(xs.astype(np.float32) - WIDTH * 0.5) * depth / focal,
            -(ys.astype(np.float32) - HEIGHT * 0.5) * depth / focal,
        ]
    ).astype(np.float32)
    cam_pos, cam_rot = camera_pose()
    world = cam_pos + local @ cam_rot.T
    return world.astype(np.float32), mask


def load_hand_pose(path, name):
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    return data["hand_poses"].item()[name]


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def pose_keyframes():
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
    return ready, hover, grasp, lift


def body_pose_at_time(t):
    ready, hover, grasp, lift = pose_keyframes()
    if t < 1.0:
        return ready, "ready"
    if t < 4.8:
        alpha = smoothstep((t - 1.0) / 3.8)
        return ready * (1.0 - alpha) + hover * alpha, "over-table-approach"
    if t < 6.4:
        alpha = smoothstep((t - 4.8) / 1.6)
        return hover * (1.0 - alpha) + grasp * alpha, "descend"
    if t < 7.4:
        return grasp, "grasp"
    alpha = smoothstep((t - 7.4) / 2.6)
    return grasp * (1.0 - alpha) + lift * alpha, "lift"


def cup_grasp_params(times, hand_poses):
    body_poses = []
    phases = []
    for t in times:
        pose, phase = body_pose_at_time(float(t))
        body_poses.append(pose)
        phases.append(phase)
    body_pose = np.asarray(body_poses, dtype=np.float32)
    frame_count = len(body_pose)

    params = {
        "betas": torch.zeros((frame_count, 10), dtype=torch.float32),
        "global_orient": torch.zeros((frame_count, 3), dtype=torch.float32),
        "body_pose": torch.as_tensor(body_pose, dtype=torch.float32),
    }
    relaxed = load_hand_pose(hand_poses, "relaxed")
    if relaxed is not None:
        if isinstance(relaxed, tuple):
            left_relaxed, right_relaxed = relaxed
        else:
            left_relaxed = right_relaxed = relaxed
        left_hand = torch.as_tensor(np.tile(left_relaxed.reshape(1, -1), (frame_count, 1)), dtype=torch.float32)
        right_hand = torch.as_tensor(np.tile(right_relaxed.reshape(1, -1), (frame_count, 1)), dtype=torch.float32)
        params["left_hand_pose"] = left_hand
        params["right_hand_pose"] = right_hand
    return params, phases


def smplx_to_gazebo_world(vertices, delta_xyz, delta_yaw):
    visual = np.column_stack(
        [vertices[:, 0], -vertices[:, 2], vertices[:, 1] + SMPLX_VISUAL_Z]
    ).astype(np.float32)
    world = visual @ rz(SMPLX_MODEL_YAW + float(delta_yaw)).T
    return world + SMPLX_MODEL_POS + np.asarray(delta_xyz, dtype=np.float32)


def smplx_to_gazebo_world_torch(vertices, delta_xyz, delta_yaw):
    visual_z = torch.tensor(SMPLX_VISUAL_Z, dtype=vertices.dtype, device=vertices.device)
    visual = torch.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1] + visual_z], dim=1)
    yaw = torch.tensor(SMPLX_MODEL_YAW, dtype=vertices.dtype, device=vertices.device) + delta_yaw
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.tensor(0.0, dtype=vertices.dtype, device=vertices.device)
    one = torch.tensor(1.0, dtype=vertices.dtype, device=vertices.device)
    rot = torch.stack(
        [
            torch.stack([c, -s, zero]),
            torch.stack([s, c, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    base_pos = torch.as_tensor(SMPLX_MODEL_POS, dtype=vertices.dtype, device=vertices.device)
    return visual @ rot.T + base_pos + delta_xyz


def fit_frame(mesh_vertices, observed_points, seed_delta, seed_yaw, max_steps):
    if len(observed_points) < 150:
        return seed_delta, seed_yaw, float("inf")

    rng = np.random.default_rng(1205)
    mesh_idx = rng.choice(len(mesh_vertices), size=min(1800, len(mesh_vertices)), replace=False)
    obs_idx = rng.choice(len(observed_points), size=min(2200, len(observed_points)), replace=False)

    mesh = torch.as_tensor(mesh_vertices[mesh_idx], dtype=torch.float32)
    obs = torch.as_tensor(observed_points[obs_idx], dtype=torch.float32)
    delta = torch.tensor(seed_delta, dtype=torch.float32, requires_grad=True)
    yaw = torch.tensor([seed_yaw], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([delta, yaw], lr=0.025)

    base_pos = torch.as_tensor(SMPLX_MODEL_POS, dtype=torch.float32)
    visual_z = torch.tensor(SMPLX_VISUAL_Z, dtype=torch.float32)
    for _ in range(max_steps):
        c = torch.cos(torch.tensor(SMPLX_MODEL_YAW, dtype=torch.float32) + yaw[0])
        s = torch.sin(torch.tensor(SMPLX_MODEL_YAW, dtype=torch.float32) + yaw[0])
        rot = torch.stack(
            [
                torch.stack([c, -s, torch.tensor(0.0)]),
                torch.stack([s, c, torch.tensor(0.0)]),
                torch.tensor([0.0, 0.0, 1.0]),
            ]
        )
        visual = torch.stack([mesh[:, 0], -mesh[:, 2], mesh[:, 1] + visual_z], dim=1)
        pred = visual @ rot.T + base_pos + delta
        distances = torch.cdist(pred, obs)
        loss = distances.min(dim=1).values.mean() + 0.35 * distances.min(dim=0).values.mean()
        loss = loss + 0.05 * torch.sum(delta * delta) + 0.03 * torch.sum(yaw * yaw)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta[:] = torch.clamp(delta, min=torch.tensor([-0.35, -0.35, -0.25]), max=torch.tensor([0.35, 0.35, 0.35]))
            yaw[:] = torch.clamp(yaw, -0.45, 0.45)
    return delta.detach().numpy(), float(yaw.detach()[0]), float(loss.detach())


def fit_pose_frame(
    model,
    base_body_pose,
    left_hand_pose,
    right_hand_pose,
    observed_points,
    seed_delta,
    seed_yaw,
    seed_pose_delta,
    max_steps,
    max_mesh_points,
):
    if len(observed_points) < 150:
        return seed_delta, seed_yaw, seed_pose_delta, None, float("inf")

    rng = np.random.default_rng(1701)
    mesh_idx = rng.choice(10475, size=min(max_mesh_points, 10475), replace=False)
    hand_extra = RIGHT_HAND_VERTEX_RANGE[:: max(1, len(RIGHT_HAND_VERTEX_RANGE) // 220)]
    mesh_idx = np.unique(np.concatenate([mesh_idx, hand_extra])).astype(np.int64)
    obs_idx = rng.choice(len(observed_points), size=min(2600, len(observed_points)), replace=False)

    obs = torch.as_tensor(observed_points[obs_idx], dtype=torch.float32)
    base_body = torch.as_tensor(base_body_pose.reshape(1, -1), dtype=torch.float32)
    delta = torch.tensor(seed_delta, dtype=torch.float32, requires_grad=True)
    yaw = torch.tensor(float(seed_yaw), dtype=torch.float32, requires_grad=True)
    pose_delta = torch.tensor(seed_pose_delta, dtype=torch.float32, requires_grad=True)
    seed_pose = torch.tensor(seed_pose_delta, dtype=torch.float32)
    optimizer = torch.optim.Adam([delta, yaw, pose_delta], lr=0.018)

    betas = torch.zeros((1, 10), dtype=torch.float32)
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    left_hand = left_hand_pose.reshape(1, -1) if left_hand_pose is not None else None
    right_hand = right_hand_pose.reshape(1, -1) if right_hand_pose is not None else None

    for _ in range(max_steps):
        body_pose = base_body.clone()
        for slot, joint_idx in enumerate(POSE_FIT_JOINTS):
            offset = joint_idx * 3
            body_pose[:, offset : offset + 3] = body_pose[:, offset : offset + 3] + pose_delta[slot * 3 : slot * 3 + 3]
        kwargs = {
            "betas": betas,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "return_verts": True,
        }
        if left_hand is not None:
            kwargs["left_hand_pose"] = left_hand
        if right_hand is not None:
            kwargs["right_hand_pose"] = right_hand
        vertices = model(**kwargs).vertices[0]
        pred = smplx_to_gazebo_world_torch(vertices[mesh_idx], delta, yaw)
        distances = torch.cdist(pred, obs)
        pred_to_obs = distances.min(dim=1).values
        obs_to_pred = distances.min(dim=0).values
        robust_pred = torch.sqrt(pred_to_obs * pred_to_obs + 0.0025)
        robust_obs = torch.sqrt(obs_to_pred * obs_to_pred + 0.0025)
        loss = robust_pred.mean() + 0.45 * robust_obs.mean()
        loss = loss + 0.025 * torch.sum(delta * delta) + 0.02 * yaw * yaw
        loss = loss + 0.035 * torch.mean(pose_delta * pose_delta) + 0.06 * torch.mean((pose_delta - seed_pose) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta[:] = torch.clamp(delta, min=torch.tensor([-0.35, -0.35, -0.25]), max=torch.tensor([0.35, 0.35, 0.35]))
            yaw.clamp_(-0.45, 0.45)
            pose_delta.clamp_(-0.95, 0.95)

    with torch.no_grad():
        body_pose = base_body.clone()
        for slot, joint_idx in enumerate(POSE_FIT_JOINTS):
            offset = joint_idx * 3
            body_pose[:, offset : offset + 3] = body_pose[:, offset : offset + 3] + pose_delta[slot * 3 : slot * 3 + 3]
        kwargs = {
            "betas": betas,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "return_verts": True,
        }
        if left_hand is not None:
            kwargs["left_hand_pose"] = left_hand
        if right_hand is not None:
            kwargs["right_hand_pose"] = right_hand
        vertices = model(**kwargs).vertices[0].detach().cpu().numpy()
    return (
        delta.detach().numpy(),
        float(yaw.detach()),
        pose_delta.detach().numpy(),
        vertices,
        float(loss.detach()),
    )


def project_world(point_world):
    cam_pos, cam_rot = camera_pose()
    local = (np.asarray(point_world, dtype=np.float32) - cam_pos) @ cam_rot
    depth = float(local[0])
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    u = WIDTH * 0.5 - focal * local[1] / max(depth, 1e-6)
    v = HEIGHT * 0.5 - focal * local[2] / max(depth, 1e-6)
    return [round(float(u), 2), round(float(v), 2)], round(depth, 4), local


def main():
    parser = argparse.ArgumentParser(
        description="Fit SMPL-X mesh / pose to D455 RGB-D frames and output the retarget trajectory contract."
    )
    parser.add_argument("--rgbd-npz", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hand-poses")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=31)
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--fit-mode", choices=["global", "pose"], default="pose")
    parser.add_argument("--mesh-points", type=int, default=1700)
    args = parser.parse_args()

    data = np.load(args.rgbd_npz)
    rgb = data["rgb"]
    depth_m = data["depth_m"]
    times = data["times"]
    if args.max_frames > 0 and len(times) > args.max_frames:
        idx = np.linspace(0, len(times) - 1, args.max_frames).round().astype(int)
        rgb = rgb[idx]
        depth_m = depth_m[idx]
        times = times[idx]

    batch_model = smplx.create(
        args.model_dir,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        batch_size=len(times),
    )
    params, phases = cup_grasp_params(times, args.hand_poses)
    pose_model = None
    if args.fit_mode == "pose":
        pose_model = smplx.create(
            args.model_dir,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            num_betas=10,
            batch_size=1,
        )
        vertices_batch = None
    else:
        with torch.no_grad():
            output = batch_model(return_verts=True, **params)
        vertices_batch = output.vertices.detach().cpu().numpy()

    rng = np.random.default_rng(455)
    delta = np.zeros(3, dtype=np.float32)
    yaw = 0.0
    pose_delta = np.zeros(len(POSE_FIT_JOINTS) * 3, dtype=np.float32)
    left_hand_pose = params.get("left_hand_pose")
    right_hand_pose = params.get("right_hand_pose")
    samples = []
    diagnostics = []
    for frame, t in enumerate(times):
        observed_points, mask = backproject_mask(rgb[frame], depth_m[frame], args.max_points, rng)
        frame_steps = args.steps if frame == 0 else max(12, args.steps // 3)
        if args.fit_mode == "pose":
            left_frame = left_hand_pose[frame : frame + 1] if left_hand_pose is not None else None
            right_frame = right_hand_pose[frame : frame + 1] if right_hand_pose is not None else None
            delta, yaw, pose_delta, fitted_vertices, loss = fit_pose_frame(
                pose_model,
                params["body_pose"][frame].detach().cpu().numpy(),
                left_frame,
                right_frame,
                observed_points,
                delta,
                yaw,
                pose_delta,
                frame_steps,
                args.mesh_points,
            )
            if fitted_vertices is None:
                with torch.no_grad():
                    fallback = pose_model(
                        betas=torch.zeros((1, 10), dtype=torch.float32),
                        global_orient=torch.zeros((1, 3), dtype=torch.float32),
                        body_pose=params["body_pose"][frame : frame + 1],
                        left_hand_pose=left_frame,
                        right_hand_pose=right_frame,
                        return_verts=True,
                    ).vertices[0].detach().cpu().numpy()
                fitted_vertices = fallback
            fitted_world = smplx_to_gazebo_world(fitted_vertices, delta, yaw)
        else:
            delta, yaw, loss = fit_frame(vertices_batch[frame], observed_points, delta, yaw, frame_steps)
            fitted_world = smplx_to_gazebo_world(vertices_batch[frame], delta, yaw)
        hand_world = fitted_world[RIGHT_HAND_VERTEX_RANGE].mean(axis=0)
        pixel, depth, hand_camera = project_world(hand_world)
        confidence = 0.0 if len(observed_points) < 150 else float(np.clip(1.0 / (1.0 + loss * 8.0), 0.05, 1.0))
        samples.append(
            {
                "time": round(float(t), 4),
                "phase": phases[frame],
                "right_hand_world": [round(float(v), 6) for v in hand_world],
                "right_hand_camera": [round(float(v), 6) for v in hand_camera],
                "pixel": pixel,
                "depth_m": depth,
                "confidence": round(confidence, 4),
            }
        )
        diagnostics.append(
            {
                "time": round(float(t), 4),
                "observed_points": int(len(observed_points)),
                "fit_loss": round(float(loss), 6),
                "fit_mode": args.fit_mode,
                "delta_xyz": [round(float(v), 6) for v in delta],
                "delta_yaw": round(float(yaw), 6),
                "pose_delta_right_arm": [round(float(v), 6) for v in pose_delta],
            }
        )

    cam_pos, _ = camera_pose()
    payload = {
        "schema": "smplx_d455_reconstructed_right_hand_trajectory/v1",
        "frame": "world",
        "source": "rgbd_smplx_pose_fitting" if args.fit_mode == "pose" else "rgbd_smplx_mesh_fitting",
        "seconds": float(times[-1]) if len(times) else 0.0,
        "fps": float(1.0 / np.median(np.diff(times))) if len(times) > 1 else 1.0,
        "camera": {
            "topic_rgb": "/tripod_d455/depth/image",
            "topic_depth": "/tripod_d455/depth/depth_image",
            "width": WIDTH,
            "height": HEIGHT,
            "hfov": HFOV,
            "position_world": [round(float(v), 6) for v in cam_pos],
        },
        "diagnostics": {
            "method": (
                "white-human RGB segmentation + raw D455 depth point cloud + "
                f"SMPL-X {args.fit_mode} chamfer fit"
            ),
            "limitation": (
                "Pose mode optimizes selected right-arm SMPL-X axis-angle parameters plus global alignment. "
                "It is still an offline local optimizer, not a learned monocular HMR model."
            ),
            "frames": diagnostics,
        },
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mean_loss = np.mean([item["fit_loss"] for item in diagnostics if math.isfinite(item["fit_loss"])])
    print(f"rgbd_smplx_{args.fit_mode}_fit_samples={len(samples)} mean_loss={mean_loss:.4f} output={out}")


if __name__ == "__main__":
    raise SystemExit(main())
