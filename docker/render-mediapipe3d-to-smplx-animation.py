#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np


WIDTH = 960
HEIGHT = 720
HFOV = 0.82
NEAR = 0.1
FAR = 8.0

SMPLX_MODEL_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
SMPLX_VISUAL_Z = 1.30
CAMERA_POS = np.array([0.0, -3.0, 1.05], dtype=np.float32)
CAMERA_TARGET = np.array([0.0, 0.0, 0.88], dtype=np.float32)

JOINTS = {
    "left_hip": 1,
    "right_hip": 2,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

POSE_SLOTS = {
    "left_shoulder": 15,
    "right_shoulder": 16,
    "right_elbow": 18,
    "right_wrist": 20,
}


def normalize(vector, fallback=None):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        if fallback is None:
            return np.zeros_like(vector)
        return normalize(fallback)
    return vector / norm


def smplx_to_world(points):
    points = np.asarray(points, dtype=np.float32)
    visual = np.column_stack([points[:, 0], -points[:, 2], points[:, 1] + SMPLX_VISUAL_Z])
    return visual + SMPLX_MODEL_POS


def smplx_forward(model, torch, body_pose, return_verts, left_hand_pose=None, right_hand_pose=None):
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
    if left_hand_pose is not None:
        params["left_hand_pose"] = left_hand_pose
    if right_hand_pose is not None:
        params["right_hand_pose"] = right_hand_pose
    return model(return_verts=return_verts, **params)


def look_at_rotation(camera_pos, target):
    forward = normalize(target - camera_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(forward, world_up), [1.0, 0.0, 0.0])
    up = normalize(np.cross(right, forward), [0.0, 0.0, 1.0])
    return np.stack([right, up, forward], axis=1).astype(np.float32)


CAMERA_ROT = look_at_rotation(CAMERA_POS, CAMERA_TARGET)


def project(points):
    local = (points - CAMERA_POS) @ CAMERA_ROT
    depth = local[:, 2]
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    u = WIDTH * 0.5 + focal * local[:, 0] / np.maximum(depth, 1e-6)
    v = HEIGHT * 0.5 - focal * local[:, 1] / np.maximum(depth, 1e-6)
    return np.column_stack([u, v]).astype(np.float32), depth.astype(np.float32)


def rasterize_mesh(image, zbuf, vertices, faces, color, shade=True):
    projected, depth = project(vertices)
    light = normalize(np.array([-0.4, -0.5, 1.0], dtype=np.float32))
    color = np.asarray(color, dtype=np.float32)
    for face in faces:
        d = depth[face]
        if np.any(d <= NEAR) or np.all(d >= FAR):
            continue
        tri = projected[face]
        min_x = max(0, int(math.floor(float(np.min(tri[:, 0])))))
        max_x = min(WIDTH - 1, int(math.ceil(float(np.max(tri[:, 0])))))
        min_y = max(0, int(math.floor(float(np.min(tri[:, 1])))))
        max_y = min(HEIGHT - 1, int(math.ceil(float(np.max(tri[:, 1])))))
        if min_x > max_x or min_y > max_y:
            continue
        p0, p1, p2 = tri
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(float(area)) < 1e-5:
            continue
        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((p1[0] - xx) * (p2[1] - yy) - (p1[1] - yy) * (p2[0] - xx)) / area
        w1 = ((p2[0] - xx) * (p0[1] - yy) - (p2[1] - yy) * (p0[0] - xx)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(inside):
            continue
        z = w0 * d[0] + w1 * d[1] + w2 * d[2]
        patch = zbuf[min_y : max_y + 1, min_x : max_x + 1]
        mask = inside & (z < patch)
        if not np.any(mask):
            continue
        intensity = 1.0
        if shade:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normalize(normal, [0.0, 0.0, 1.0])
            intensity = 0.52 + 0.48 * max(0.0, float(np.dot(normal, light)))
        patch[mask] = z[mask]
        image_patch = image[min_y : max_y + 1, min_x : max_x + 1]
        image_patch[mask] = np.clip(color * intensity, 0, 255).astype(np.uint8)


def draw_point(image, zbuf, point, color, radius=8):
    projected, depth = project(point.reshape(1, 3))
    if depth[0] <= NEAR or depth[0] >= FAR:
        return
    x, y = projected[0]
    x0, x1 = max(0, int(x - radius)), min(WIDTH - 1, int(x + radius))
    y0, y1 = max(0, int(y - radius)), min(HEIGHT - 1, int(y + radius))
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
    patch = zbuf[y0 : y1 + 1, x0 : x1 + 1]
    visible = mask & (depth[0] <= patch + 0.02)
    patch[visible] = depth[0]
    image_patch = image[y0 : y1 + 1, x0 : x1 + 1]
    image_patch[visible] = color


def draw_line(image, zbuf, a, b, color, steps=80):
    for alpha in np.linspace(0.0, 1.0, steps):
        point = a * (1.0 - alpha) + b * alpha
        draw_point(image, zbuf, point, color, radius=4)


def render_background(image):
    image[:] = [223, 226, 223]
    floor_y = int(HEIGHT * 0.62)
    image[floor_y:, :] = [186, 187, 181]
    for y in range(floor_y, HEIGHT):
        blend = (y - floor_y) / max(1, HEIGHT - floor_y)
        image[y, :, :] = np.array([186, 187, 181]) * (1.0 - 0.12 * blend)


def write_ppm(path, image):
    with path.open("wb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def load_hand_pose(path, name):
    if path is None:
        return None, None
    data = np.load(path, allow_pickle=True)
    return data["hand_poses"].item()[name]


def body_basis_from(points):
    ls = points["left_shoulder"]
    rs = points["right_shoulder"]
    lh = points["left_hip"]
    rh = points["right_hip"]
    x_axis = normalize(rs - ls, [1.0, 0.0, 0.0])
    up_axis = normalize((ls + rs) * 0.5 - (lh + rh) * 0.5, [0.0, 1.0, 0.0])
    z_axis = normalize(np.cross(x_axis, up_axis), [0.0, 0.0, 1.0])
    up_axis = normalize(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    return x_axis, up_axis, z_axis


def extract_targets(skeleton, neutral_joints, fps):
    samples = skeleton["samples"]
    names = ["left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_hip", "right_hip"]
    neutral = {name: neutral_joints[JOINTS[name]].astype(np.float32) for name in names}
    smplx_x, smplx_up, smplx_z = body_basis_from(neutral)
    smplx_rs = neutral["right_shoulder"]
    upper_len = float(np.linalg.norm(neutral["right_elbow"] - neutral["right_shoulder"]))
    lower_len = float(np.linalg.norm(neutral["right_wrist"] - neutral["right_elbow"]))
    target_upper = []
    target_lower = []
    confidences = []
    times = []
    for sample in samples:
        mp = {name: np.asarray(sample["landmarks_world"][name], dtype=np.float32) for name in names}
        mp_x, mp_up, mp_z = body_basis_from(mp)
        upper = normalize(mp["right_elbow"] - mp["right_shoulder"], [0.0, -1.0, 0.0])
        lower = normalize(mp["right_wrist"] - mp["right_elbow"], [0.0, -1.0, 0.0])

        def remap(direction):
            coords = np.array([np.dot(direction, mp_x), np.dot(direction, mp_up), np.dot(direction, mp_z)], dtype=np.float32)
            return normalize(coords[0] * smplx_x + coords[1] * smplx_up + coords[2] * smplx_z, [1.0, 0.0, 0.0])

        target_upper.append(remap(upper) * upper_len)
        target_lower.append(remap(lower) * lower_len)
        confidences.append(float(sample.get("confidence", 1.0)))
        times.append(float(sample.get("time", len(times) / fps)))
    return smplx_rs, np.asarray(target_upper, dtype=np.float32), np.asarray(target_lower, dtype=np.float32), np.asarray(confidences), times


def fit_body_pose(model, skeleton, fps, iterations=260):
    import torch

    frame_count = len(skeleton["samples"])
    zero_pose = torch.zeros((frame_count, 63), dtype=torch.float32)
    with torch.no_grad():
        neutral_output = smplx_forward(model, torch, zero_pose, return_verts=False)
    neutral_joints = neutral_output.joints[0].detach().cpu().numpy().astype(np.float32)
    shoulder_anchor, target_upper, target_lower, confidences, times = extract_targets(skeleton, neutral_joints, fps)

    body_base = torch.zeros((frame_count, 63), dtype=torch.float32)
    # Keep the non-driving left arm relaxed down so the right-arm motion is readable.
    body_base[:, POSE_SLOTS["left_shoulder"] * 3 : POSE_SLOTS["left_shoulder"] * 3 + 3] = torch.tensor([0.0, 0.0, -1.25])
    drive_slots = [POSE_SLOTS["right_shoulder"], POSE_SLOTS["right_elbow"], POSE_SLOTS["right_wrist"]]
    drive = torch.zeros((frame_count, len(drive_slots), 3), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([drive], lr=0.045)
    upper_target = torch.as_tensor(target_upper, dtype=torch.float32)
    lower_target = torch.as_tensor(target_lower, dtype=torch.float32)
    weights = torch.as_tensor(np.clip(confidences, 0.25, 1.0), dtype=torch.float32).view(-1, 1)

    for _ in range(iterations):
        pose = body_base.clone()
        for idx, slot in enumerate(drive_slots):
            pose[:, slot * 3 : slot * 3 + 3] = drive[:, idx, :]
        output = smplx_forward(model, torch, pose, return_verts=False)
        joints = output.joints
        rs = joints[:, JOINTS["right_shoulder"], :]
        re = joints[:, JOINTS["right_elbow"], :]
        rw = joints[:, JOINTS["right_wrist"], :]
        pred_upper = re - rs
        pred_lower = rw - re
        loss_upper = torch.mean(weights * (pred_upper - upper_target) ** 2)
        loss_lower = torch.mean(weights * (pred_lower - lower_target) ** 2)
        reg = 0.0035 * torch.mean(drive**2)
        smooth = 0.0
        if frame_count > 1:
            smooth = 0.030 * torch.mean((drive[1:] - drive[:-1]) ** 2)
        loss = loss_upper + loss_lower + reg + smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            drive.clamp_(-1.65, 1.65)

    pose = body_base.clone()
    with torch.no_grad():
        for idx, slot in enumerate(drive_slots):
            pose[:, slot * 3 : slot * 3 + 3] = drive[:, idx, :]
        output = smplx_forward(model, torch, pose, return_verts=True)
    diagnostics = {
        "samples": frame_count,
        "confidence_mean": round(float(np.mean(confidences)), 4),
        "confidence_min": round(float(np.min(confidences)), 4),
        "fit_note": "Optimized SMPL-X right shoulder, right elbow, and right wrist axis-angle pose against MediaPipe 3D right-arm bone directions.",
    }
    return pose.detach().cpu().numpy().astype(np.float32), output.vertices.detach().cpu().numpy().astype(np.float32), output.joints.detach().cpu().numpy().astype(np.float32), diagnostics


def render_video(vertices_by_frame, joints_by_frame, faces, out_dir, fps, keep_frames=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("frame_*.ppm"):
        old.unlink()
    for frame_idx, vertices in enumerate(vertices_by_frame):
        world_vertices = smplx_to_world(vertices)
        joints = smplx_to_world(joints_by_frame[frame_idx])
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        zbuf = np.full((HEIGHT, WIDTH), FAR, dtype=np.float32)
        render_background(image)
        rasterize_mesh(image, zbuf, world_vertices, faces, [242, 242, 238], shade=True)
        # Draw the fitted right-arm skeleton on top to make the applied MediaPipe motion explicit.
        arm_color = np.array([0, 110, 255], dtype=np.uint8)
        draw_line(image, zbuf, joints[JOINTS["right_shoulder"]], joints[JOINTS["right_elbow"]], arm_color)
        draw_line(image, zbuf, joints[JOINTS["right_elbow"]], joints[JOINTS["right_wrist"]], arm_color)
        draw_point(image, zbuf, joints[JOINTS["right_wrist"]], np.array([220, 35, 20], dtype=np.uint8), radius=8)
        write_ppm(frames_dir / f"frame_{frame_idx:04d}.ppm", image)
    video_path = out_dir / "taichi_mediapipe3d_smplx_animation.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not keep_frames:
        for frame_file in frames_dir.glob("frame_*.ppm"):
            frame_file.unlink()
        frames_dir.rmdir()
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Fit MediaPipe 3D right-arm landmarks to a continuous SMPL-X mesh animation.")
    parser.add_argument("--skeleton-json", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--hand-poses", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--iterations", type=int, default=260)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    try:
        import smplx
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Activate .venv-smplx with smplx and torch.") from exc

    skeleton = json.loads(args.skeleton_json.read_text(encoding="utf-8"))
    frame_count = len(skeleton["samples"])
    model = smplx.create(
        str(args.model_dir),
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        batch_size=frame_count,
    )
    body_pose, vertices, joints, diagnostics = fit_body_pose(model, skeleton, args.fps, iterations=args.iterations)
    left_hand, right_hand = load_hand_pose(args.hand_poses, "relaxed")
    if left_hand is not None or right_hand is not None:
        left_hand_pose = None
        right_hand_pose = None
        if left_hand is not None:
            left_hand_pose = torch.as_tensor(np.tile(left_hand.reshape(1, -1), (frame_count, 1)), dtype=torch.float32)
        if right_hand is not None:
            right_hand_pose = torch.as_tensor(np.tile(right_hand.reshape(1, -1), (frame_count, 1)), dtype=torch.float32)
        with torch.no_grad():
            output = smplx_forward(
                model,
                torch,
                torch.as_tensor(body_pose, dtype=torch.float32),
                return_verts=True,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
            )
        vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
        joints = output.joints.detach().cpu().numpy().astype(np.float32)
    video_path = render_video(vertices, joints, model.faces.astype(np.int64), args.out_dir, args.fps, keep_frames=args.keep_frames)
    np.save(args.out_dir / "smplx_pose_fit_body_pose.npy", body_pose)
    manifest = {
        "video": video_path.name,
        "skeleton_json": str(args.skeleton_json),
        "fps": args.fps,
        "frames": frame_count,
        "schema": "taichi_mediapipe3d_to_smplx_animation/v1",
        "diagnostics": diagnostics,
        "driven_joints": ["right_shoulder", "right_elbow", "right_wrist"],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(video_path)


if __name__ == "__main__":
    main()
