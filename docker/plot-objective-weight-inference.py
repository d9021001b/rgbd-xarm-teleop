#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def score(row):
    return (
        row["forearm_mean_deg"]
        + 1.15 * row["upper_mean_deg"]
        + 0.12 * row["forearm_max_deg"]
        + 0.12 * row["upper_max_deg"]
        + 0.85 * row["included_mean_deg"]
        + 0.08 * row["included_max_deg"]
        + 2500.0 * max(0.0, row["position_mean_m"] - 0.004)
        + 6500.0 * max(0.0, row["position_max_m"] - 0.020)
    )


def optimizer_row(base, label, rel, forearm_weight, upper_weight, included_weight, fps):
    item = json.loads((base / rel / "objective_sweep_results.json").read_text(encoding="utf-8"))[0]
    return {
        "label": label,
        "fps": fps,
        "forearm_weight": forearm_weight,
        "upper_arm_weight": upper_weight,
        "included_angle_weight": included_weight,
        "position_mean_m": item["position_mean_m"],
        "position_max_m": item["position_max_m"],
        "forearm_mean_deg": item["forearm_mean_deg"],
        "forearm_max_deg": item["forearm_max_deg"],
        "upper_mean_deg": item["upper_mean_deg"],
        "upper_max_deg": item["upper_max_deg"],
        "included_mean_deg": item["included_mean_deg"],
        "included_max_deg": item["included_max_deg"],
        "score": item["score"],
        "source_dir": str(base / rel),
    }


def analyzer_row(base, label, rel, forearm_weight, upper_weight, included_weight, fps):
    summary = json.loads((base / rel / "trajectory_comparison_summary.json").read_text(encoding="utf-8"))
    row = {
        "label": label,
        "fps": fps,
        "forearm_weight": forearm_weight,
        "upper_arm_weight": upper_weight,
        "included_angle_weight": included_weight,
        "position_mean_m": summary["position_raw_error_m"]["mean"],
        "position_max_m": summary["position_raw_error_m"]["max"],
        "forearm_mean_deg": summary["forearm_projected_abs_error_deg"]["mean_max_projection"],
        "forearm_max_deg": summary["forearm_projected_abs_error_deg"]["max_projection"],
        "upper_mean_deg": summary["upper_arm_projected_abs_error_deg"]["mean_max_projection"],
        "upper_max_deg": summary["upper_arm_projected_abs_error_deg"]["max_projection"],
        "included_mean_deg": summary["upper_lower_included_angle_abs_error_deg"]["mean"],
        "included_max_deg": summary["upper_lower_included_angle_abs_error_deg"]["max"],
        "source_dir": str(base / rel),
    }
    row["score"] = score(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        analyzer_row(base, "baseline_original_config_10fps", "trajectory_comparison_baseline_recomputed", None, None, None, 10),
        analyzer_row(
            base,
            "rotbest_no_included_10fps",
            "objective_optimization_rotbest_ultralow_10fps/trajectory_comparison",
            0.08,
            0.04,
            0.0,
            10,
        ),
        optimizer_row(base, "inc004_5fps", "objective_optimization_inc004_5fps", 0.08, 0.04, 0.04, 5),
        optimizer_row(base, "inc008_5fps", "objective_optimization_inc008_5fps", 0.08, 0.04, 0.08, 5),
        optimizer_row(base, "inc012_5fps", "objective_optimization_inc012_5fps", 0.08, 0.04, 0.12, 5),
        optimizer_row(base, "inc020_5fps", "objective_optimization_inc020_5fps", 0.08, 0.04, 0.20, 5),
        analyzer_row(
            base,
            "best_inc012_10fps",
            "objective_optimization_inc012_10fps/trajectory_comparison",
            0.08,
            0.04,
            0.12,
            10,
        ),
    ]

    fields = [
        "label",
        "fps",
        "forearm_weight",
        "upper_arm_weight",
        "included_angle_weight",
        "position_mean_m",
        "position_max_m",
        "forearm_mean_deg",
        "forearm_max_deg",
        "upper_mean_deg",
        "upper_max_deg",
        "included_mean_deg",
        "included_max_deg",
        "score",
        "source_dir",
    ]
    with (out / "objective_weight_inference_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "objective_weight_inference_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    sweep = [
        row
        for row in rows
        if isinstance(row["included_angle_weight"], (int, float))
        and row["label"] != "rotbest_no_included_10fps"
        and row["fps"] == 5
    ]
    sweep = sorted(sweep, key=lambda row: (row["included_angle_weight"], row["fps"]))
    validation = next(row for row in rows if row["label"] == "best_inc012_10fps")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    x_values = [row["included_angle_weight"] for row in sweep]
    ax1.plot(x_values, [row["included_mean_deg"] for row in sweep], marker="o", label="included angle mean error (deg)")
    ax1.plot(x_values, [row["forearm_mean_deg"] for row in sweep], marker="o", label="forearm projected mean-max error (deg)")
    ax1.plot(x_values, [row["upper_mean_deg"] for row in sweep], marker="o", label="upper projected mean-max error (deg)")
    ax1.scatter(
        [validation["included_angle_weight"]],
        [validation["included_mean_deg"]],
        marker="*",
        s=180,
        color="tab:blue",
        edgecolors="black",
        zorder=5,
        label="10fps validation included error",
    )
    ax1.set_xlabel("included_angle_weight")
    ax1.set_ylabel("angle error (deg)")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        x_values,
        [1000.0 * row["position_mean_m"] for row in sweep],
        color="tab:red",
        marker="s",
        linestyle="--",
        label="TCP mean error (mm)",
    )
    ax2.scatter(
        [validation["included_angle_weight"]],
        [1000.0 * validation["position_mean_m"]],
        marker="*",
        s=180,
        color="tab:red",
        edgecolors="black",
        zorder=5,
        label="10fps validation TCP mean (mm)",
    )
    ax2.plot(
        x_values,
        [1000.0 * row["position_max_m"] for row in sweep],
        color="tab:pink",
        marker="s",
        linestyle="--",
        label="TCP max error (mm)",
    )
    ax2.set_ylabel("position error (mm)")

    lines, labels = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines_2, labels + labels_2, loc="best")
    fig.suptitle("IK Objective Weight Inference Tradeoff")
    fig.tight_layout()
    fig.savefig(out / "objective_weight_tradeoff.png", dpi=180)


if __name__ == "__main__":
    raise SystemExit(main())
