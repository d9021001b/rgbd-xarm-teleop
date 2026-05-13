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


PROJECTION_LABELS = ["XY yaw atan2(y,x)", "XZ pitch atan2(z,x)", "YZ pitch atan2(z,y)"]


def load_retarget_module(path):
    spec = importlib.util.spec_from_file_location("retarget_smplx_xarm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_angles(values):
    values = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(values), np.cos(values))


def normed(vector, fallback=(1.0, 0.0, 0.0)):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float)
    return vector / norm


def angle_between_deg(a, b):
    a = normed(a)
    b = normed(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def projected_angles(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray(
        [
            math.atan2(y, x),
            math.atan2(z, x),
            math.atan2(z, y),
        ],
        dtype=float,
    )


def similarity_fit(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    src_mu = source.mean(axis=0)
    tgt_mu = target.mean(axis=0)
    src_centered = source - src_mu
    tgt_centered = target - tgt_mu
    covariance = (tgt_centered.T @ src_centered) / max(1, len(source))
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ correction) / max(variance, 1e-12))
    translation = tgt_mu - scale * (rotation @ src_mu)
    fitted = (scale * (rotation @ source.T)).T + translation
    return fitted, {"scale": scale, "rotation": rotation, "translation": translation}


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


def load_comparison_samples(args):
    retarget = load_retarget_module(args.retarget_script)
    config = json.loads(Path(args.retarget_config).read_text(encoding="utf-8"))
    retarget.apply_retarget_config(config)

    urdf_text = get_urdf_text(retarget, args.urdf_text)
    chain = retarget.KinematicChain(
        urdf_text,
        base_link=retarget.BASE_LINK,
        tip_link=retarget.TIP_LINK,
    )

    human_trajectory = retarget.load_reconstructed_trajectory(args.smplx_json)
    xarm_payload = json.loads(Path(args.xarm_json).read_text(encoding="utf-8"))
    seconds = min(float(xarm_payload.get("seconds", human_trajectory[-1]["time"])), human_trajectory[-1]["time"])
    fps = float(args.fps or xarm_payload.get("fps", 10.0))
    times = np.arange(0.0, seconds + 0.5 / fps, 1.0 / fps)
    q_samples = resample_robot_trajectory(xarm_payload, times)

    shoulder_index = retarget.robot_shoulder_joint_index()
    if shoulder_index is None:
        raise RuntimeError("retarget config must define functional.robot_shoulder_joint_index")
    elbow_index = retarget.robot_elbow_joint_index()
    source_rotation = retarget.functional_source_vector_rotation_base()

    samples = []
    diagnostics = xarm_payload.get("diagnostics", [])
    for sample_idx, (t, q) in enumerate(zip(times, q_samples)):
        point_world, point_base, phase, arm = retarget.sample_reconstructed_target(human_trajectory, float(t))
        raw_wrist_base = (
            retarget.world_to_robot_base(point_world)
            if point_base is None
            else np.asarray(point_base, dtype=float)
        )
        functional_wrist_base = retarget.clip_base_target(
            retarget.conditioned_base_target(raw_wrist_base, float(t))
        )
        target_wrist_base = functional_wrist_base
        diagnostic = diagnostics[min(sample_idx, len(diagnostics) - 1)] if diagnostics else {}
        if diagnostic.get("target") is not None:
            target_wrist_base = np.asarray(diagnostic["target"], dtype=float)
            phase = diagnostic.get("phase", phase)
        elbow_base = retarget.global_workspace_transform_base(retarget.world_to_robot_base(arm["right_elbow_world"]))
        shoulder_base = retarget.global_workspace_transform_base(retarget.world_to_robot_base(arm["right_shoulder_world"]))

        transform, joint_positions, _ = chain.fk(q)
        tcp_base = transform[:3, 3]
        robot_shoulder = joint_positions[shoulder_index]
        robot_elbow = joint_positions[elbow_index]

        human_upper = source_rotation @ (elbow_base - shoulder_base)
        human_forearm = source_rotation @ (functional_wrist_base - elbow_base)
        robot_upper = robot_elbow - robot_shoulder
        robot_forearm = tcp_base - robot_elbow

        robot_forearm_proj = projected_angles(robot_forearm)
        robot_upper_proj = projected_angles(robot_upper)
        functional_metrics = diagnostic.get("functional_metrics", {})
        if functional_metrics.get("forearm_projected_rad") is not None:
            forearm_projected_error = np.asarray(functional_metrics["forearm_projected_rad"], dtype=float)
            human_forearm_proj = wrap_angles(robot_forearm_proj + forearm_projected_error)
        else:
            human_forearm_proj = projected_angles(human_forearm)
            forearm_projected_error = wrap_angles(human_forearm_proj - robot_forearm_proj)
        if functional_metrics.get("upper_arm_projected_rad") is not None:
            upper_projected_error = np.asarray(functional_metrics["upper_arm_projected_rad"], dtype=float)
            human_upper_proj = wrap_angles(robot_upper_proj + upper_projected_error)
        else:
            human_upper_proj = projected_angles(human_upper)
            upper_projected_error = wrap_angles(human_upper_proj - robot_upper_proj)

        samples.append(
            {
                "time": float(t),
                "phase": phase,
                "smplx_wrist_base": target_wrist_base,
                "smplx_functional_wrist_base": functional_wrist_base,
                "xarm_tcp_base": tcp_base,
                "position_abs_error": np.abs(tcp_base - target_wrist_base),
                "position_error_norm": float(np.linalg.norm(tcp_base - target_wrist_base)),
                "smplx_elbow_angle_deg": angle_between_deg(human_upper, human_forearm),
                "xarm_elbow_angle_deg": angle_between_deg(robot_upper, robot_forearm),
                "smplx_forearm_projected_deg": np.degrees(human_forearm_proj),
                "xarm_forearm_projected_deg": np.degrees(robot_forearm_proj),
                "forearm_projected_abs_error_deg": np.abs(np.degrees(forearm_projected_error)),
                "smplx_upper_projected_deg": np.degrees(human_upper_proj),
                "xarm_upper_projected_deg": np.degrees(robot_upper_proj),
                "upper_projected_abs_error_deg": np.abs(np.degrees(upper_projected_error)),
                "robot_shoulder_base": robot_shoulder,
                "robot_elbow_base": robot_elbow,
                "smplx_shoulder_base": shoulder_base,
                "smplx_elbow_base": elbow_base,
            }
        )
    return samples, config


def as_array(samples, key):
    return np.asarray([sample[key] for sample in samples], dtype=float)


def save_position_plots(out_dir, times, smplx_pos, xarm_pos, fitted_pos):
    axis_labels = ["X base (m)", "Y base (m)", "Z base (m)"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for idx, label in enumerate(axis_labels):
        axes[idx].plot(times, smplx_pos[:, idx], label="SMPL-X right hand target", linewidth=2)
        axes[idx].plot(times, xarm_pos[:, idx], label="xArm7 TCP", linewidth=2, linestyle="--")
        axes[idx].plot(times, fitted_pos[:, idx], label="SMPL-X fitted to TCP", linewidth=1.5, linestyle=":")
        axes[idx].set_ylabel(label)
        axes[idx].grid(True, alpha=0.25)
    raw_error = np.linalg.norm(xarm_pos - smplx_pos, axis=1)
    fit_error = np.linalg.norm(xarm_pos - fitted_pos, axis=1)
    axes[3].plot(times, raw_error, label="raw |TCP - target|", linewidth=2)
    axes[3].plot(times, fit_error, label="similarity-fit |TCP - fitted target|", linewidth=2)
    axes[3].set_ylabel("error (m)")
    axes[3].set_xlabel("source trajectory time (s)")
    axes[3].grid(True, alpha=0.25)
    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle("Position Trajectory: SMPL-X Right Hand vs xArm7 TCP")
    fig.tight_layout()
    fig.savefig(out_dir / "position_trajectory_and_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    projection_specs = [
        (0, 1, "X base (m)", "Y base (m)", "XY projection"),
        (0, 2, "X base (m)", "Z base (m)", "XZ projection"),
        (1, 2, "Y base (m)", "Z base (m)", "YZ projection"),
    ]
    for ax, (a, b, xlabel, ylabel, title) in zip(axes, projection_specs):
        ax.plot(smplx_pos[:, a], smplx_pos[:, b], label="SMPL-X target", linewidth=2)
        ax.plot(xarm_pos[:, a], xarm_pos[:, b], label="xArm7 TCP", linewidth=2)
        ax.plot(fitted_pos[:, a], fitted_pos[:, b], label="SMPL-X fitted", linewidth=1.5, linestyle=":")
        ax.scatter(smplx_pos[0, a], smplx_pos[0, b], marker="o", label="start")
        ax.scatter(smplx_pos[-1, a], smplx_pos[-1, b], marker="x", label="end")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Position Trajectory Fit Projections")
    fig.tight_layout()
    fig.savefig(out_dir / "position_trajectory_3d_fit.png", dpi=180)
    plt.close(fig)


def save_elbow_angle_plot(out_dir, times, smplx_angle, xarm_angle):
    error = np.abs(smplx_angle - xarm_angle)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(times, smplx_angle, label="SMPL-X upper/lower arm included angle", linewidth=2)
    axes[0].plot(times, xarm_angle, label="xArm7 upper/forearm included angle", linewidth=2, linestyle="--")
    axes[0].set_ylabel("angle (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(times, error, label="absolute error", linewidth=2, color="tab:red")
    axes[1].set_ylabel("abs error (deg)")
    axes[1].set_xlabel("source trajectory time (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.suptitle("Upper-vs-Lower Arm Spatial Included Angle")
    fig.tight_layout()
    fig.savefig(out_dir / "upper_lower_arm_included_angle_error.png", dpi=180)
    plt.close(fig)


def save_projected_angle_plot(out_dir, filename, title, times, smplx_angles, xarm_angles, abs_error):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for idx, label in enumerate(PROJECTION_LABELS):
        axes[idx, 0].plot(times, smplx_angles[:, idx], label="SMPL-X", linewidth=2)
        axes[idx, 0].plot(times, xarm_angles[:, idx], label="xArm7", linewidth=2, linestyle="--")
        axes[idx, 0].set_ylabel(f"{label}\nangle (deg)")
        axes[idx, 0].grid(True, alpha=0.25)
        axes[idx, 1].plot(times, abs_error[:, idx], label="abs error", linewidth=2, color="tab:red")
        axes[idx, 1].set_ylabel("abs error (deg)")
        axes[idx, 1].grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("source trajectory time (s)")
    axes[-1, 1].set_xlabel("source trajectory time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def save_summary(out_dir, samples, fit_info, fitted_pos):
    times = as_array(samples, "time")
    smplx_pos = as_array(samples, "smplx_wrist_base")
    xarm_pos = as_array(samples, "xarm_tcp_base")
    pos_error = np.linalg.norm(xarm_pos - smplx_pos, axis=1)
    fit_error = np.linalg.norm(xarm_pos - fitted_pos, axis=1)
    elbow_error = np.abs(as_array(samples, "smplx_elbow_angle_deg") - as_array(samples, "xarm_elbow_angle_deg"))
    forearm_error = as_array(samples, "forearm_projected_abs_error_deg")
    upper_error = as_array(samples, "upper_projected_abs_error_deg")

    summary = {
        "samples": len(samples),
        "time_start_seconds": float(times[0]),
        "time_end_seconds": float(times[-1]),
        "position_raw_error_m": {
            "mean": float(np.mean(pos_error)),
            "median": float(np.median(pos_error)),
            "max": float(np.max(pos_error)),
        },
        "position_similarity_fit_error_m": {
            "mean": float(np.mean(fit_error)),
            "median": float(np.median(fit_error)),
            "max": float(np.max(fit_error)),
        },
        "upper_lower_included_angle_abs_error_deg": {
            "mean": float(np.mean(elbow_error)),
            "median": float(np.median(elbow_error)),
            "max": float(np.max(elbow_error)),
        },
        "forearm_projected_abs_error_deg": {
            "mean_xyz_projection": np.mean(forearm_error, axis=0).tolist(),
            "max_xyz_projection": np.max(forearm_error, axis=0).tolist(),
            "mean_max_projection": float(np.mean(np.max(forearm_error, axis=1))),
            "max_projection": float(np.max(forearm_error)),
        },
        "upper_arm_projected_abs_error_deg": {
            "mean_xyz_projection": np.mean(upper_error, axis=0).tolist(),
            "max_xyz_projection": np.max(upper_error, axis=0).tolist(),
            "mean_max_projection": float(np.mean(np.max(upper_error, axis=1))),
            "max_projection": float(np.max(upper_error)),
        },
        "similarity_fit": {
            "scale": float(fit_info["scale"]),
            "rotation": fit_info["rotation"].tolist(),
            "translation": fit_info["translation"].tolist(),
        },
        "projection_labels": PROJECTION_LABELS,
        "robot_mapping": {
            "tcp": "link_tcp",
            "upper_arm": "joint_positions[robot_shoulder_joint_index=2] -> joint_positions[robot_elbow_joint_index=4]",
            "forearm": "joint_positions[robot_elbow_joint_index=4] -> link_tcp",
        },
    }
    (out_dir / "trajectory_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out_dir / "trajectory_comparison_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "phase",
                "pos_error_norm_m",
                "fit_pos_error_norm_m",
                "elbow_angle_error_deg",
                "forearm_xy_error_deg",
                "forearm_xz_error_deg",
                "forearm_yz_error_deg",
                "upper_xy_error_deg",
                "upper_xz_error_deg",
                "upper_yz_error_deg",
            ]
        )
        for idx, sample in enumerate(samples):
            writer.writerow(
                [
                    sample["time"],
                    sample["phase"],
                    float(pos_error[idx]),
                    float(fit_error[idx]),
                    float(elbow_error[idx]),
                    *[float(v) for v in forearm_error[idx]],
                    *[float(v) for v in upper_error[idx]],
                ]
            )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx-json", required=True)
    parser.add_argument("--xarm-json", required=True)
    parser.add_argument("--retarget-config", required=True)
    parser.add_argument("--retarget-script", default="/tmp/retarget-smplx-hand-to-xarm.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--urdf-text", help="Optional file containing robot_description text.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, _ = load_comparison_samples(args)

    times = as_array(samples, "time")
    smplx_pos = as_array(samples, "smplx_wrist_base")
    xarm_pos = as_array(samples, "xarm_tcp_base")
    fitted_pos, fit_info = similarity_fit(smplx_pos, xarm_pos)

    save_position_plots(out_dir, times, smplx_pos, xarm_pos, fitted_pos)
    save_elbow_angle_plot(
        out_dir,
        times,
        as_array(samples, "smplx_elbow_angle_deg"),
        as_array(samples, "xarm_elbow_angle_deg"),
    )
    save_projected_angle_plot(
        out_dir,
        "forearm_projected_angles_error.png",
        "Forearm / Lower-Arm Projected Spatial Angles",
        times,
        as_array(samples, "smplx_forearm_projected_deg"),
        as_array(samples, "xarm_forearm_projected_deg"),
        as_array(samples, "forearm_projected_abs_error_deg"),
    )
    save_projected_angle_plot(
        out_dir,
        "upper_arm_projected_angles_error.png",
        "Upper Arm / Shoulder-to-Elbow Projected Spatial Angles",
        times,
        as_array(samples, "smplx_upper_projected_deg"),
        as_array(samples, "xarm_upper_projected_deg"),
        as_array(samples, "upper_projected_abs_error_deg"),
    )
    summary = save_summary(out_dir, samples, fit_info, fitted_pos)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
