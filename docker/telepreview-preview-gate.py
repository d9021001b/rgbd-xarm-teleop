#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GATE_CONFIG = {
    "schema": "telepreview_preview_gate/v1",
    "preview_gate": {
        "mode": "preview_then_align",
        "min_tcp_z_base": 0.09,
        "hard_gates": {
            "max_tcp_error_m": 0.03,
            "min_table_clearance_m": 0.0,
            "min_joint_limit_margin_rad": 0.0,
            "max_joint_step_p95_rad": 0.65,
        },
        "soft_gates": {
            "min_joint_limit_margin_warn_rad": 0.025,
            "max_joint_step_abs_rad": 0.90,
            "max_joint_velocity_p95_rad_s": 6.0,
            "max_joint_acceleration_p95_rad_s2": 85.0,
            "max_joint_jerk_rms_rad_s3": 500.0,
            "max_included_angle_mean_deg": 8.0,
            "max_forearm_mean_max_deg": 35.0,
            "max_upper_arm_mean_max_deg": 45.0,
        },
    },
}


def deep_merge(base, update):
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json(path, default=None):
    if path is None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_retarget_module(path):
    spec = importlib.util.spec_from_file_location("retarget_smplx_xarm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_urdf_text(retarget, urdf_text_path=None):
    if urdf_text_path:
        return retarget.clean_robot_description(Path(urdf_text_path).read_text(encoding="utf-8").strip())
    try:
        return retarget.fetch_robot_description_with_cli()
    except Exception:
        fallback = Path("/tmp/xarm_robot_description_param.txt")
        if fallback.exists():
            return retarget.clean_robot_description(fallback.read_text(encoding="utf-8").strip())
        raise


def resample_robot_trajectory(payload, times):
    positions = np.asarray(payload["positions"], dtype=float)
    seconds = float(payload.get("seconds", times[-1]))
    fps = float(payload.get("fps", (len(positions) - 1) / max(seconds, 1e-9)))
    source_times = np.arange(len(positions), dtype=float) / fps
    source_times = np.clip(source_times, 0.0, seconds)
    result = np.empty((len(times), positions.shape[1]), dtype=float)
    for axis in range(positions.shape[1]):
        result[:, axis] = np.interp(times, source_times, positions[:, axis])
    return result


def finite_joint_limit_margin(q, limits):
    margins = []
    for value, (lower, upper) in zip(q, limits):
        if math.isfinite(lower):
            margins.append(float(value - lower))
        if math.isfinite(upper):
            margins.append(float(upper - value))
    return min(margins) if margins else math.inf


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def max_or_zero(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(values)) if values.size else 0.0


def mean_or_zero(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else 0.0


def make_gate(name, kind, value, threshold, relation):
    if relation == "<=":
        passed = value <= threshold
    elif relation == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"Unsupported relation: {relation}")
    return {
        "name": name,
        "kind": kind,
        "value": float(value),
        "threshold": float(threshold),
        "relation": relation,
        "pass": bool(passed),
    }


def save_samples_csv(out_dir, rows):
    path = out_dir / "telepreview_gate_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "phase",
                "tcp_error_m",
                "tcp_z_base",
                "table_clearance_m",
                "joint_limit_margin_rad",
                "max_joint_step_rad",
                "included_angle_abs_error_deg",
                "forearm_max_deg",
                "upper_arm_max_deg",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def save_gate_plot(out_dir, rows, summary, gate_config):
    times = np.asarray([row["time"] for row in rows], dtype=float)
    tcp_error = np.asarray([row["tcp_error_m"] for row in rows], dtype=float)
    tcp_z = np.asarray([row["tcp_z_base"] for row in rows], dtype=float)
    joint_step = np.asarray([row["max_joint_step_rad"] for row in rows], dtype=float)
    included = np.asarray([row["included_angle_abs_error_deg"] for row in rows], dtype=float)
    forearm = np.asarray([row["forearm_max_deg"] for row in rows], dtype=float)
    upper = np.asarray([row["upper_arm_max_deg"] for row in rows], dtype=float)

    hard = gate_config["preview_gate"]["hard_gates"]
    soft = gate_config["preview_gate"]["soft_gates"]
    min_tcp_z = float(gate_config["preview_gate"]["min_tcp_z_base"])

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(times, 1000.0 * tcp_error, linewidth=2, label="TCP preview error")
    axes[0].axhline(1000.0 * float(hard["max_tcp_error_m"]), color="tab:red", linestyle="--", label="hard gate")
    axes[0].set_ylabel("TCP error (mm)")
    axes[0].legend(loc="best")

    axes[1].plot(times, tcp_z, linewidth=2, label="TCP z in base")
    axes[1].axhline(min_tcp_z, color="tab:red", linestyle="--", label="min allowed z")
    axes[1].set_ylabel("z (m)")
    axes[1].legend(loc="best")

    axes[2].plot(times, joint_step, linewidth=2, label="max joint step/frame")
    axes[2].axhline(float(hard["max_joint_step_p95_rad"]), color="tab:orange", linestyle="--", label="p95 hard gate")
    axes[2].axhline(float(soft["max_joint_step_abs_rad"]), color="tab:red", linestyle=":", label="abs soft gate")
    axes[2].set_ylabel("rad/frame")
    axes[2].legend(loc="best")

    axes[3].plot(times, included, linewidth=2, label="included angle")
    axes[3].plot(times, forearm, linewidth=1.8, label="forearm max projection")
    axes[3].plot(times, upper, linewidth=1.8, label="upper-arm max projection")
    axes[3].set_ylabel("error (deg)")
    axes[3].set_xlabel("preview time (s)")
    axes[3].legend(loc="best", ncol=3)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"TelePreview Gate Overview: {summary['decision']}")
    fig.tight_layout()
    fig.savefig(out_dir / "telepreview_gate_overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="TelePreview-style preview gate for xArm retarget trajectories.")
    parser.add_argument("--xarm-json", required=True)
    parser.add_argument("--retarget-config", required=True)
    parser.add_argument("--gate-config")
    parser.add_argument("--comparison-summary", help="Optional direct-retarget analyzer summary to embed in the report.")
    parser.add_argument("--retarget-script", default="/tmp/retarget-smplx-hand-to-xarm.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--urdf-text", help="Optional file containing robot_description text.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_config = deep_merge(DEFAULT_GATE_CONFIG, load_json(args.gate_config, {}))
    retarget_config = load_json(args.retarget_config)
    retarget = load_retarget_module(args.retarget_script)
    retarget.apply_retarget_config(retarget_config)

    urdf_text = get_urdf_text(retarget, args.urdf_text)
    chain = retarget.KinematicChain(urdf_text, base_link=retarget.BASE_LINK, tip_link=retarget.TIP_LINK)

    payload = load_json(args.xarm_json)
    positions = np.asarray(payload["positions"], dtype=float)
    fps = float(args.fps or payload.get("fps", 10.0))
    seconds = float(payload.get("seconds", (len(positions) - 1) / max(fps, 1e-9)))
    times = np.arange(0.0, seconds + 0.5 / fps, 1.0 / fps)
    q_samples = resample_robot_trajectory(payload, times)
    diagnostics = payload.get("diagnostics", [])

    min_tcp_z = float(gate_config["preview_gate"]["min_tcp_z_base"])
    rows = []
    for idx, (time_s, q) in enumerate(zip(times, q_samples)):
        transform, _, _ = chain.fk(q)
        tcp = transform[:3, 3]
        diagnostic = diagnostics[min(idx, len(diagnostics) - 1)] if diagnostics else {}
        target = np.asarray(diagnostic.get("target", tcp), dtype=float)
        tcp_error = float(diagnostic.get("error", np.linalg.norm(tcp - target)))
        functional = diagnostic.get("functional_metrics", {}) or {}
        row = {
            "time": float(time_s),
            "phase": diagnostic.get("phase", "preview"),
            "tcp_error_m": tcp_error,
            "tcp_z_base": float(tcp[2]),
            "table_clearance_m": float(tcp[2] - min_tcp_z),
            "joint_limit_margin_rad": finite_joint_limit_margin(q, chain.limits),
            "max_joint_step_rad": 0.0,
            "included_angle_abs_error_deg": float(functional.get("included_angle_abs_error_deg", 0.0)),
            "forearm_max_deg": float(functional.get("forearm_max_deg", 0.0)),
            "upper_arm_max_deg": float(functional.get("upper_arm_max_deg", 0.0)),
        }
        rows.append(row)

    joint_steps = np.max(np.abs(np.diff(q_samples, axis=0)), axis=1) if len(q_samples) > 1 else np.asarray([])
    for idx, step in enumerate(joint_steps, start=1):
        rows[idx]["max_joint_step_rad"] = float(step)
    velocities = joint_steps * fps
    acceleration_values = (
        np.max(np.abs(np.diff(np.diff(q_samples, axis=0) * fps, axis=0) * fps), axis=1)
        if len(q_samples) > 2
        else np.asarray([])
    )
    jerk_values = (
        np.diff(np.diff(np.diff(q_samples, axis=0) * fps, axis=0) * fps, axis=0) * fps
        if len(q_samples) > 3
        else np.asarray([])
    )

    tcp_errors = np.asarray([row["tcp_error_m"] for row in rows], dtype=float)
    table_clearances = np.asarray([row["table_clearance_m"] for row in rows], dtype=float)
    joint_margins = np.asarray([row["joint_limit_margin_rad"] for row in rows], dtype=float)
    included = np.asarray([row["included_angle_abs_error_deg"] for row in rows], dtype=float)
    forearm = np.asarray([row["forearm_max_deg"] for row in rows], dtype=float)
    upper = np.asarray([row["upper_arm_max_deg"] for row in rows], dtype=float)

    hard_cfg = gate_config["preview_gate"]["hard_gates"]
    soft_cfg = gate_config["preview_gate"]["soft_gates"]
    gates = [
        make_gate("tcp_error_max_m", "hard", max_or_zero(tcp_errors), hard_cfg["max_tcp_error_m"], "<="),
        make_gate("table_clearance_min_m", "hard", float(np.min(table_clearances)), hard_cfg["min_table_clearance_m"], ">="),
        make_gate(
            "joint_limit_margin_min_rad",
            "hard",
            float(np.min(joint_margins[np.isfinite(joint_margins)])) if np.any(np.isfinite(joint_margins)) else math.inf,
            hard_cfg["min_joint_limit_margin_rad"],
            ">=",
        ),
        make_gate(
            "joint_step_p95_rad",
            "hard",
            percentile(joint_steps, 95),
            hard_cfg["max_joint_step_p95_rad"],
            "<=",
        ),
        make_gate(
            "joint_limit_margin_warn_rad",
            "soft",
            float(np.min(joint_margins[np.isfinite(joint_margins)])) if np.any(np.isfinite(joint_margins)) else math.inf,
            soft_cfg["min_joint_limit_margin_warn_rad"],
            ">=",
        ),
        make_gate("joint_step_abs_rad", "soft", max_or_zero(joint_steps), soft_cfg["max_joint_step_abs_rad"], "<="),
        make_gate(
            "joint_velocity_p95_rad_s",
            "soft",
            percentile(velocities, 95),
            soft_cfg["max_joint_velocity_p95_rad_s"],
            "<=",
        ),
        make_gate(
            "joint_acceleration_p95_rad_s2",
            "soft",
            percentile(acceleration_values, 95),
            soft_cfg["max_joint_acceleration_p95_rad_s2"],
            "<=",
        ),
        make_gate(
            "joint_jerk_rms_rad_s3",
            "soft",
            float(np.sqrt(np.mean(np.square(jerk_values)))) if jerk_values.size else 0.0,
            soft_cfg["max_joint_jerk_rms_rad_s3"],
            "<=",
        ),
        make_gate(
            "included_angle_mean_deg",
            "soft",
            mean_or_zero(included),
            soft_cfg["max_included_angle_mean_deg"],
            "<=",
        ),
        make_gate(
            "forearm_mean_max_deg",
            "soft",
            mean_or_zero(forearm),
            soft_cfg["max_forearm_mean_max_deg"],
            "<=",
        ),
        make_gate(
            "upper_arm_mean_max_deg",
            "soft",
            mean_or_zero(upper),
            soft_cfg["max_upper_arm_mean_max_deg"],
            "<=",
        ),
    ]

    hard_pass = all(gate["pass"] for gate in gates if gate["kind"] == "hard")
    soft_pass = all(gate["pass"] for gate in gates if gate["kind"] == "soft")
    decision = "APPROVE" if hard_pass and soft_pass else "APPROVE_WITH_WARNINGS" if hard_pass else "BLOCK"
    summary = {
        "schema": "telepreview_preview_gate_result/v1",
        "paper_reference": gate_config.get("paper_reference", "TelePreview preview-then-align mode"),
        "xarm_json": args.xarm_json,
        "retarget_config": args.retarget_config,
        "gate_config": args.gate_config or "builtin_defaults",
        "samples": len(rows),
        "time_start_seconds": float(times[0]),
        "time_end_seconds": float(times[min(len(times), len(rows)) - 1]),
        "decision": decision,
        "hard_pass": hard_pass,
        "soft_pass": soft_pass,
        "gates": gates,
        "preview_metrics": {
            "tcp_error_mean_m": mean_or_zero(tcp_errors),
            "tcp_error_max_m": max_or_zero(tcp_errors),
            "tcp_z_min_base_m": float(np.min([row["tcp_z_base"] for row in rows])),
            "table_clearance_min_m": float(np.min(table_clearances)),
            "joint_step_abs_max_rad": max_or_zero(joint_steps),
            "joint_step_p95_rad": percentile(joint_steps, 95),
            "joint_velocity_p95_rad_s": percentile(velocities, 95),
            "joint_acceleration_p95_rad_s2": percentile(acceleration_values, 95),
            "joint_jerk_rms_rad_s3": float(np.sqrt(np.mean(np.square(jerk_values)))) if jerk_values.size else 0.0,
            "included_angle_mean_deg": mean_or_zero(included),
            "forearm_mean_max_deg": mean_or_zero(forearm),
            "upper_arm_mean_max_deg": mean_or_zero(upper),
        },
        "telepreview_mapping": {
            "preview_mode": "Render/evaluate the retarget trajectory in the Gazebo digital twin without executing it.",
            "gate_mode": "Block execution if hard safety gates fail; approve with warnings if only soft quality gates fail.",
            "align_mode": "After approval, the final preview trajectory can be sent to the robot or to MoveIt2 for retimed execution.",
        },
    }
    if args.comparison_summary:
        comparison = load_json(args.comparison_summary)
        summary["direct_hybrid_comparison"] = {
            "source": args.comparison_summary,
            "position_raw_error_m": comparison.get("position_raw_error_m"),
            "upper_lower_included_angle_abs_error_deg": comparison.get("upper_lower_included_angle_abs_error_deg"),
            "forearm_projected_abs_error_deg": comparison.get("forearm_projected_abs_error_deg"),
            "upper_arm_projected_abs_error_deg": comparison.get("upper_arm_projected_abs_error_deg"),
        }

    save_samples_csv(out_dir, rows)
    save_gate_plot(out_dir, rows, summary, gate_config)
    (out_dir / "telepreview_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
