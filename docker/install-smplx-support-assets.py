#!/usr/bin/env python3
import argparse
import shutil
import zipfile
from pathlib import Path


def extract_zip(zip_path, out_dir):
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise SystemExit(f"zip not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename)
            if any(part == "__MACOSX" for part in name.parts):
                continue
            if name.name == ".DS_Store" or name.name.startswith("._"):
                continue
            archive.extract(member, out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Install local SMPL-X support assets such as VPoser and MANO/FLAME correspondences."
    )
    parser.add_argument("--vposer-zip", help="Path to vposer_v1_0.zip.")
    parser.add_argument(
        "--correspondences-zip",
        help="Path to smplx_mano_flame_correspondences.zip.",
    )
    parser.add_argument(
        "--out-dir",
        default="xarm_ros2/xarm_gazebo/models/smplx_operator/support",
        help="Target support asset directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.vposer_zip:
        print(f"vposer={extract_zip(args.vposer_zip, out_dir / 'vposer')}")
    if args.correspondences_zip:
        print(f"correspondences={extract_zip(args.correspondences_zip, out_dir / 'correspondences')}")


if __name__ == "__main__":
    main()
