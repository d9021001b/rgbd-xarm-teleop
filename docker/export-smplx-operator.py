#!/usr/bin/env python3
import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def read_uv_template(source):
    if source is None:
        return None, None, None

    source_path = Path(source).expanduser().resolve()
    temp_dir = None
    if source_path.suffix.lower() == ".zip":
        temp_dir = tempfile.TemporaryDirectory()
        extract_dir = Path(temp_dir.name)
        with zipfile.ZipFile(source_path) as archive:
            archive.extractall(extract_dir)
        objs = sorted(extract_dir.rglob("*.obj"))
        textures = sorted(extract_dir.rglob("*.png")) + sorted(extract_dir.rglob("*.jpg"))
        if not objs:
            raise SystemExit(f"No OBJ found in UV template zip: {source_path}")
        source_path = objs[0]
        texture_path = textures[0] if textures else None
    else:
        texture_path = None

    uvs = []
    face_uvs = []
    for line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("vt "):
            parts = line.split()
            uvs.append((float(parts[1]), float(parts[2])))
        elif line.startswith("f "):
            uv_face = []
            for token in line.split()[1:]:
                pieces = token.split("/")
                uv_face.append(int(pieces[1]) if len(pieces) > 1 and pieces[1] else 0)
            face_uvs.append(tuple(uv_face))
    return uvs, face_uvs, (texture_path, temp_dir)


def write_mtl(path, texture_name):
    path.write_text(
        "\n".join(
            [
                "newmtl smplx_uv_material",
                "Ka 1.000 1.000 1.000",
                "Kd 1.000 1.000 1.000",
                "Ks 0.050 0.050 0.050",
                "Ns 10.000",
                "d 1.000",
                f"map_Kd {texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_obj(path, vertices, faces, uvs=None, face_uvs=None, texture_path=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = None
    texture_name = None
    if texture_path is not None:
        texture_path = Path(texture_path)
        texture_name = texture_path.name
        (path.parent / texture_name).write_bytes(texture_path.read_bytes())
        mtl_name = "smplx_operator.mtl"
        write_mtl(path.parent / mtl_name, texture_name)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated from a locally licensed SMPL-X model\n")
        if mtl_name:
            handle.write(f"mtllib {mtl_name}\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f}\n")
        if uvs:
            for u, v in uvs:
                handle.write(f"vt {u:.7f} {v:.7f}\n")
        if mtl_name:
            handle.write("usemtl smplx_uv_material\n")
        for idx, face in enumerate(faces):
            a, b, c = face + 1
            if face_uvs and idx < len(face_uvs) and len(face_uvs[idx]) == 3:
                ua, ub, uc = face_uvs[idx]
                handle.write(f"f {a}/{ua} {b}/{ub} {c}/{uc}\n")
            else:
                handle.write(f"f {a} {b} {c}\n")


def load_params(path):
    if path is None:
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def load_named_hand_pose(path, name):
    if path is None:
        return None, None

    data = np.load(path, allow_pickle=True)
    hand_poses = data["hand_poses"].item()
    if name not in hand_poses:
        available = ", ".join(sorted(hand_poses))
        raise SystemExit(f"Hand pose '{name}' not found. Available hand poses: {available}")
    return hand_poses[name]


def preset_params(name, left_hand_pose=None, right_hand_pose=None):
    params = {}
    if name == "neutral":
        return params
    if name != "cup-grasp":
        raise SystemExit(f"Unsupported preset: {name}")

    body_pose = np.zeros((1, 63), dtype=np.float32)
    # Body pose uses SMPL-X body joints after the root joint. These values lift
    # the right arm forward and inward so the hand can be aligned with the cup,
    # while the non-grasping arm is lowered to avoid a distracting T-pose.
    body_pose[0, 15 * 3 : 15 * 3 + 3] = [0.0, 0.0, -1.40]  # left shoulder down
    body_pose[0, 16 * 3 : 16 * 3 + 3] = [0.0, 0.0, 0.70]   # right shoulder reach
    body_pose[0, 18 * 3 : 18 * 3 + 3] = [0.0, 0.0, -0.60]  # right elbow
    body_pose[0, 20 * 3 : 20 * 3 + 3] = [0.0, 0.0, 0.16]   # right wrist
    params["body_pose"] = body_pose

    if left_hand_pose is not None:
        params["left_hand_pose"] = np.asarray(left_hand_pose, dtype=np.float32).reshape(1, -1)
    if right_hand_pose is not None:
        params["right_hand_pose"] = np.asarray(right_hand_pose, dtype=np.float32).reshape(1, -1)
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Export a licensed local SMPL-X body model to a Gazebo OBJ mesh."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing SMPL-X model files.")
    parser.add_argument("--out", required=True, help="Output OBJ path.")
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--params", help="Optional .npz file with SMPL-X parameters from HMR.")
    parser.add_argument(
        "--preset",
        default="neutral",
        choices=["neutral", "cup-grasp"],
        help="Built-in static pose preset used when --params does not provide a value.",
    )
    parser.add_argument(
        "--hand-poses",
        help="Optional smplx_handposes.npz from the SMPL-X Blender addon.",
    )
    parser.add_argument(
        "--hand-pose-name",
        default="relaxed",
        help="Named hand pose to use from --hand-poses for the cup-grasp preset.",
    )
    parser.add_argument("--uv-template", help="Optional UV template OBJ or zip, such as smplx_uv_2023.zip.")
    parser.add_argument("--texture", help="Optional texture image. Overrides texture found in --uv-template zip.")
    args = parser.parse_args()

    try:
        import torch
        import smplx
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: python3 -m pip install smplx torch numpy"
        ) from exc

    params = load_params(args.params)
    left_hand_pose, right_hand_pose = load_named_hand_pose(args.hand_poses, args.hand_pose_name)
    defaults = preset_params(args.preset, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
    defaults.update(params)
    params = defaults
    model = smplx.create(
        args.model_dir,
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        batch_size=1,
    )

    tensor_params = {}
    for key, value in params.items():
        if key in {
            "global_orient",
            "body_pose",
            "betas",
            "left_hand_pose",
            "right_hand_pose",
            "expression",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "transl",
        }:
            tensor_params[key] = torch.as_tensor(value, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad():
        output = model(return_verts=True, **tensor_params)

    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int64)
    uvs, face_uvs, texture_info = read_uv_template(args.uv_template)
    texture_path = Path(args.texture).expanduser().resolve() if args.texture else None
    temp_dir = None
    if texture_path is None and texture_info is not None:
        texture_path, temp_dir = texture_info
    write_obj(args.out, vertices, faces, uvs=uvs, face_uvs=face_uvs, texture_path=texture_path)
    if temp_dir is not None:
        temp_dir.cleanup()
    print(args.out)


if __name__ == "__main__":
    main()
