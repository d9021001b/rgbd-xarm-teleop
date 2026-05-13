#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np


WIDTH = 848
HEIGHT = 480
HFOV = 2.05
NEAR = 0.1
FAR = 8.0

TRIPOD_POS = np.array([-1.20, -3.10, 0.0], dtype=np.float32)
TRIPOD_RPY = np.array([0.0, 0.0, 1.12], dtype=np.float32)
SENSOR_POS = np.array([0.08, 0.0, 1.44], dtype=np.float32)
SENSOR_RPY = np.array([0.0, 0.12, 0.0], dtype=np.float32)

SMPLX_MODEL_POS = np.array([-0.283, -1.34, 0.0], dtype=np.float32)
SMPLX_MODEL_YAW = 3.55
SMPLX_VISUAL_Z = 1.26

CUP_HOME = np.array([0.34, -0.93, 1.065], dtype=np.float32)


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


def cup_position(frame, fps, hand_center):
    t = frame / fps
    if t < 7.4:
        return CUP_HOME.copy()
    alpha = smoothstep((t - 7.4) / 2.6)
    carried = hand_center + np.array([0.026, 0.002, -0.070], dtype=np.float32)
    return CUP_HOME * (1.0 - alpha) + carried * alpha


def smplx_to_world(vertices):
    visual_vertices = np.column_stack(
        [vertices[:, 0], -vertices[:, 2], vertices[:, 1] + SMPLX_VISUAL_Z]
    ).astype(np.float32)
    world = visual_vertices @ rz(SMPLX_MODEL_YAW).T
    return world + SMPLX_MODEL_POS


def camera_transform():
    tripod_r = rpy_matrix(TRIPOD_RPY)
    sensor_r = tripod_r @ rpy_matrix(SENSOR_RPY)
    sensor_pos = TRIPOD_POS + tripod_r @ SENSOR_POS
    return sensor_pos.astype(np.float32), sensor_r.astype(np.float32)


def project(points, camera_pos, camera_rot):
    local = (points - camera_pos) @ camera_rot
    depth = local[:, 0]
    focal = WIDTH / (2.0 * math.tan(HFOV / 2.0))
    u = WIDTH * 0.5 - focal * local[:, 1] / np.maximum(depth, 1e-6)
    v = HEIGHT * 0.5 - focal * local[:, 2] / np.maximum(depth, 1e-6)
    return np.column_stack([u, v]).astype(np.float32), depth.astype(np.float32)


def rasterize_mesh(image, zbuf, vertices, faces, color, camera_pos, camera_rot, shade=True):
    projected, depth = project(vertices, camera_pos, camera_rot)
    light = np.array([-0.5, -0.7, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
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

        if shade:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal /= norm
            intensity = 0.55 + 0.45 * max(0.0, float(np.dot(normal, light)))
        else:
            intensity = 1.0
        patch[mask] = z[mask]
        image_patch = image[min_y : max_y + 1, min_x : max_x + 1]
        image_patch[mask] = np.clip(color * intensity, 0, 255).astype(np.uint8)


def box_mesh(center, size):
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, dtype=np.float32) * 0.5
    vertices = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def cylinder_mesh(center, radius=0.036, height=0.105, segments=32):
    vertices = []
    cx, cy, cz = center
    z0 = cz - height * 0.5
    z1 = cz + height * 0.5
    for z in (z0, z1):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            vertices.append([cx + radius * math.cos(a), cy + radius * math.sin(a), z])
    vertices.append([cx, cy, z0])
    vertices.append([cx, cy, z1])
    bottom = 2 * segments
    top = bottom + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([bottom, j, i])
        faces.append([top, segments + i, segments + j])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def render_environment(image, zbuf, camera_pos, camera_rot):
    table_v, table_f = box_mesh([0.0, -0.84, 1.00], [1.20, 0.72, 0.055])
    rasterize_mesh(image, zbuf, table_v, table_f, [82, 52, 28], camera_pos, camera_rot)
    for x in (-0.50, 0.50):
        for y in (-1.12, -0.56):
            leg_v, leg_f = box_mesh([x, y, 0.50], [0.035, 0.035, 1.00])
            rasterize_mesh(image, zbuf, leg_v, leg_f, [38, 38, 38], camera_pos, camera_rot)


def render_floor_and_walls(image):
    image[:] = [216, 219, 216]
    horizon = int(HEIGHT * 0.44)
    image[horizon:, :] = [176, 179, 176]
    for y in range(horizon, HEIGHT):
        blend = (y - horizon) / max(1, HEIGHT - horizon)
        image[y, :, :] = np.array([176, 179, 176]) * (1 - 0.18 * blend)


def write_ppm(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Render a 10 second SMPL-X cup grasp sequence from the D455 camera viewpoint."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hand-poses")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import smplx
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Activate .venv-smplx with smplx, torch, numpy.") from exc

    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = int(round(args.seconds * args.fps))
    body_poses, phases = build_body_poses(frame_count, args.fps)
    left_hand, right_hand = load_hand_pose(args.hand_poses, "relaxed")

    model = smplx.create(
        args.model_dir,
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        batch_size=frame_count,
    )
    tensor_params = {
        "body_pose": torch.as_tensor(body_poses, dtype=torch.float32),
    }
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

    right_hand_indices = np.arange(7331, 8129, dtype=np.int64)
    camera_pos, camera_rot = camera_transform()

    for frame in range(frame_count):
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        zbuf = np.full((HEIGHT, WIDTH), FAR, dtype=np.float32)
        render_floor_and_walls(image)
        render_environment(image, zbuf, camera_pos, camera_rot)

        smplx_world = smplx_to_world(vertices_by_frame[frame])
        hand_center = smplx_world[right_hand_indices].mean(axis=0)
        cup_center = cup_position(frame, args.fps, hand_center)
        cup_v, cup_f = cylinder_mesh(cup_center)
        rasterize_mesh(image, zbuf, cup_v, cup_f, [226, 20, 16], camera_pos, camera_rot, shade=True)
        rasterize_mesh(image, zbuf, smplx_world, faces, [245, 245, 240], camera_pos, camera_rot, shade=True)
        write_ppm(frames_dir / f"frame_{frame:04d}.ppm", image)

    rgb_video = out_dir / "d455_smplx_grasp_rgb.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "frame_%04d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(rgb_video),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    manifest = {
        "rgb_video": rgb_video.name,
        "seconds": args.seconds,
        "fps": args.fps,
        "frames": frame_count,
        "camera": "tripod_d455 simulated viewpoint",
        "motion": ["ready", "over-table-approach", "descend", "grasp", "lift"],
        "note": "Rendered from SMPL-X pose frames with the same D455 pose/FOV used in the Gazebo scene. The hand path keeps table clearance before descending onto the cup.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.keep_frames:
        for frame_file in frames_dir.glob("*.ppm"):
            frame_file.unlink()
        frames_dir.rmdir()
    print(rgb_video)


if __name__ == "__main__":
    main()
