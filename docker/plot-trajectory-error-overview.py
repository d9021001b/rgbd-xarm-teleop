#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    time = column(rows, "time")
    position_mm = 1000.0 * column(rows, "pos_error_norm_m")
    included = column(rows, "elbow_angle_error_deg")
    forearm = np.vstack(
        [
            column(rows, "forearm_xy_error_deg"),
            column(rows, "forearm_xz_error_deg"),
            column(rows, "forearm_yz_error_deg"),
        ]
    ).T
    upper = np.vstack(
        [
            column(rows, "upper_xy_error_deg"),
            column(rows, "upper_xz_error_deg"),
            column(rows, "upper_yz_error_deg"),
        ]
    ).T

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(time, position_mm, linewidth=2, color="tab:blue", label="|TCP - SMPL-X right hand|")
    axes[0].set_ylabel("position error (mm)")
    axes[0].legend(loc="best")

    axes[1].plot(time, included, linewidth=2, color="tab:red", label="upper-lower included angle")
    axes[1].set_ylabel("abs error (deg)")
    axes[1].legend(loc="best")

    labels = ["XY", "XZ", "YZ"]
    for idx, label in enumerate(labels):
        axes[2].plot(time, forearm[:, idx], linewidth=1.8, label=f"forearm {label}")
    axes[2].plot(time, np.max(forearm, axis=1), linewidth=2.2, color="black", linestyle="--", label="forearm max")
    axes[2].set_ylabel("abs error (deg)")
    axes[2].legend(loc="best", ncol=4)

    for idx, label in enumerate(labels):
        axes[3].plot(time, upper[:, idx], linewidth=1.8, label=f"upper arm {label}")
    axes[3].plot(time, np.max(upper, axis=1), linewidth=2.2, color="black", linestyle="--", label="upper arm max")
    axes[3].set_ylabel("abs error (deg)")
    axes[3].set_xlabel("SMPL-X source trajectory time (s)")
    axes[3].legend(loc="best", ncol=4)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("SMPL-X Right Arm vs xArm7 Retarget Absolute Error Overview")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    raise SystemExit(main())
