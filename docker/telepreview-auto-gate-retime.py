#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(args):
    completed = subprocess.run(args, text=True)
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(args)}")
    return completed.returncode


def load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Run TelePreview gate, automatically apply prepose+retiming for APPROVE_WITH_WARNINGS, and re-gate."
    )
    parser.add_argument("--xarm-json", required=True)
    parser.add_argument("--retarget-config", required=True)
    parser.add_argument("--gate-config", required=True)
    parser.add_argument("--comparison-summary")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--retarget-script", default="/tmp/retarget-smplx-hand-to-xarm.py")
    parser.add_argument("--gate-script", default="/tmp/telepreview-preview-gate.py")
    parser.add_argument("--retime-script", default="/tmp/telepreview-retime-trajectory.py")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-step-rad", type=float, default=0.45)
    parser.add_argument("--soft-limit-margin-rad", type=float, default=0.015)
    parser.add_argument("--prepose-hold-seconds", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    original_gate_dir = out_dir / "original_gate"
    retimed_gate_dir = out_dir / "retimed_gate"
    retimed_json = out_dir / "xarm7_telepreview_retimed_trajectory.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    common_gate_args = [
        sys.executable,
        args.gate_script,
        "--retarget-config",
        args.retarget_config,
        "--gate-config",
        args.gate_config,
        "--retarget-script",
        args.retarget_script,
        "--fps",
        str(args.fps),
    ]
    if args.comparison_summary:
        common_gate_args.extend(["--comparison-summary", args.comparison_summary])

    run_command(
        common_gate_args
        + [
            "--xarm-json",
            args.xarm_json,
            "--out-dir",
            str(original_gate_dir),
        ]
    )
    original_summary = load_summary(original_gate_dir / "telepreview_gate_summary.json")
    final_summary = original_summary
    retime_summary = None

    if original_summary["decision"] == "APPROVE_WITH_WARNINGS":
        run_command(
            [
                sys.executable,
                args.retime_script,
                "--xarm-json",
                args.xarm_json,
                "--retarget-config",
                args.retarget_config,
                "--retarget-script",
                args.retarget_script,
                "--out",
                str(retimed_json),
                "--max-step-rad",
                str(args.max_step_rad),
                "--soft-limit-margin-rad",
                str(args.soft_limit_margin_rad),
                "--prepose-hold-seconds",
                str(args.prepose_hold_seconds),
            ]
        )
        retime_payload = load_summary(retimed_json)
        retime_summary = retime_payload.get("telepreview_retiming")
        run_command(
            common_gate_args
            + [
                "--xarm-json",
                str(retimed_json),
                "--out-dir",
                str(retimed_gate_dir),
            ]
        )
        final_summary = load_summary(retimed_gate_dir / "telepreview_gate_summary.json")

    auto_summary = {
        "schema": "telepreview_auto_gate_retime_result/v1",
        "input_xarm_json": args.xarm_json,
        "retimed_xarm_json": str(retimed_json) if retime_summary is not None else None,
        "original_decision": original_summary["decision"],
        "final_decision": final_summary["decision"],
        "retime_applied": retime_summary is not None,
        "retime_summary": retime_summary,
        "original_gate_summary": str(original_gate_dir / "telepreview_gate_summary.json"),
        "final_gate_summary": str(
            (retimed_gate_dir if retime_summary is not None else original_gate_dir) / "telepreview_gate_summary.json"
        ),
    }
    (out_dir / "telepreview_auto_gate_retime_summary.json").write_text(
        json.dumps(auto_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(auto_summary, indent=2))
    return 0 if final_summary["decision"] == "APPROVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
