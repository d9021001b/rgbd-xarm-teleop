#!/usr/bin/env python3
import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


TEXTURE_KEYS = {
    "map_Ka",
    "map_Kd",
    "map_Ks",
    "map_Ke",
    "map_Bump",
    "map_bump",
    "bump",
    "disp",
    "decal",
    "norm",
    "map_Pr",
    "map_Pm",
    "map_Ps",
    "map_Ns",
    "map_d",
}


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def parse_mtllibs(obj_text):
    libs = []
    for line in obj_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            libs.extend(part for part in stripped.split()[1:] if part)
    return libs


def parse_texture_refs(mtl_text):
    refs = []
    for line in mtl_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] in TEXTURE_KEYS and len(parts) > 1:
            refs.append(parts[-1])
    return refs


def rewrite_obj_mtllibs(obj_text, copied_mtls):
    if not copied_mtls:
        return obj_text
    mtllib_line = "mtllib " + " ".join(path.name for path in copied_mtls)
    lines = []
    replaced = False
    for line in obj_text.splitlines():
        if line.strip().startswith("mtllib "):
            if not replaced:
                lines.append(mtllib_line)
                replaced = True
            continue
        lines.append(line)
    if not replaced:
        lines.insert(0, mtllib_line)
    return "\n".join(lines) + "\n"


def add_material_to_obj(obj_text, mtl_name, material_name):
    lines = []
    has_mtllib = any(line.strip().startswith("mtllib ") for line in obj_text.splitlines())
    has_usemtl = any(line.strip().startswith("usemtl ") for line in obj_text.splitlines())
    if not has_mtllib:
        lines.append(f"mtllib {mtl_name}")
    inserted_usemtl = False
    for line in obj_text.splitlines():
        if not has_usemtl and not inserted_usemtl and line.startswith("f "):
            lines.append(f"usemtl {material_name}")
            inserted_usemtl = True
        lines.append(line)
    return "\n".join(lines) + "\n"


def rewrite_mtl_texture_paths(mtl_text):
    lines = []
    for line in mtl_text.splitlines():
        parts = line.strip().split()
        if parts and parts[0] in TEXTURE_KEYS and len(parts) > 1:
            line = re.sub(r"(\S+)$", lambda match: Path(match.group(1)).name, line)
        lines.append(line)
    return "\n".join(lines) + "\n"


def find_source_assets(source):
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() != ".zip":
        return source, [source.parent], None

    temp_dir = tempfile.TemporaryDirectory()
    extract_dir = Path(temp_dir.name)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(extract_dir)
    objs = sorted(extract_dir.rglob("*.obj"))
    if not objs:
        temp_dir.cleanup()
        raise SystemExit(f"No OBJ file found in zip: {source}")
    return objs[0], [objs[0].parent], temp_dir


def write_default_mtl(path, texture_name):
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


def main():
    parser = argparse.ArgumentParser(
        description="Install OBJ/MTL/UV texture assets into the Gazebo SMPL-X operator model."
    )
    parser.add_argument("source", help="Source OBJ file, or a zip containing an OBJ and UV texture image.")
    parser.add_argument(
        "--model-dir",
        default="xarm_ros2/xarm_gazebo/models/smplx_operator",
        help="Target Gazebo model directory.",
    )
    parser.add_argument(
        "--texture-dir",
        action="append",
        default=[],
        help="Additional directory to search for texture images referenced by MTL files.",
    )
    args = parser.parse_args()

    obj_path, source_dirs, temp_dir = find_source_assets(args.source)
    if not obj_path.exists():
        raise SystemExit(f"OBJ file not found: {obj_path}")

    model_dir = Path(args.model_dir).resolve()
    mesh_dir = model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = source_dirs + [Path(path).expanduser().resolve() for path in args.texture_dir]
    obj_text = obj_path.read_text(encoding="utf-8", errors="replace")
    copied_mtls = []
    copied_textures = []

    for mtl_ref in parse_mtllibs(obj_text):
        mtl_src = next((directory / mtl_ref for directory in search_dirs if (directory / mtl_ref).exists()), None)
        if mtl_src is None:
            print(f"warning: MTL not found: {mtl_ref}")
            continue
        mtl_text = mtl_src.read_text(encoding="utf-8", errors="replace")
        for tex_ref in parse_texture_refs(mtl_text):
            tex_src = next((directory / tex_ref for directory in search_dirs if (directory / tex_ref).exists()), None)
            if tex_src is None:
                print(f"warning: texture not found: {tex_ref}")
                continue
            copied_textures.append(copy_file(tex_src, mesh_dir / tex_src.name))
        mtl_dst = mesh_dir / mtl_src.name
        mtl_dst.write_text(rewrite_mtl_texture_paths(mtl_text), encoding="utf-8")
        copied_mtls.append(mtl_dst)

    if not copied_mtls:
        texture_candidates = []
        for directory in search_dirs:
            texture_candidates.extend(sorted(directory.glob("*.png")))
            texture_candidates.extend(sorted(directory.glob("*.jpg")))
            texture_candidates.extend(sorted(directory.glob("*.jpeg")))
        if texture_candidates:
            texture_src = texture_candidates[0]
            texture_dst = copy_file(texture_src, mesh_dir / texture_src.name)
            copied_textures.append(texture_dst)
            mtl_dst = mesh_dir / "smplx_operator.mtl"
            write_default_mtl(mtl_dst, texture_dst.name)
            copied_mtls.append(mtl_dst)
            obj_text = add_material_to_obj(obj_text, mtl_dst.name, "smplx_uv_material")
    else:
        obj_text = rewrite_obj_mtllibs(obj_text, copied_mtls)

    obj_dst = mesh_dir / "smplx_operator.obj"
    obj_dst.write_text(obj_text, encoding="utf-8")

    print(f"obj={obj_dst}")
    for mtl in copied_mtls:
        print(f"mtl={mtl}")
    for texture in copied_textures:
        print(f"texture={texture}")
    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
