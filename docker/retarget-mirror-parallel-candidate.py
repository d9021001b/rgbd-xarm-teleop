#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np


DEFAULT_MIRROR_CONFIG = {
    "schema": "mirror_parallel_candidate/v1",
    "mirror": {
        "paper_reference": "MIRROR: Visual Motion Imitation via Real-time Retargeting and TeleOperation",
        "alphas": [0.25, 0.4, 0.55, 0.7, 0.85, 1.0],
        "parallel_workers": 6,
        "candidate_iterations": 10,
        "primary_damping": 0.045,
        "secondary_damping": 0.075,
        "position_tolerance": 0.014,
        "forearm_tolerance_deg": 70.0,
        "upper_arm_tolerance_deg": 85.0,
        "forearm_weight": 0.08,
        "upper_arm_weight": 0.04,
        "included_angle_weight": 0.12,
        "secondary_step_scale": 0.45,
        "max_step": 0.085,
        "finite_difference_step": 1e-4,
        "secondary_position_guard": 0.045,
        "lyapunov_eta": 0.002,
        "position_score_weight": 2.0,
        "forearm_score_weight": 1.0,
        "upper_arm_score_weight": 0.8,
        "included_angle_score_weight": 0.8,
        "joint_update_score_weight": 0.04,
        "anti_self_score_weight": 0.25,
        "min_tcp_z_base": 0.09,
        "min_joint_margin_rad": 0.0,
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


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def finite_joint_margin(q, limits):
    margins = []
    for value, (lower, upper) in zip(q, limits):
        if math.isfinite(lower):
            margins.append(float(value - lower))
        if math.isfinite(upper):
            margins.append(float(upper - value))
    return min(margins) if margins else math.inf


def smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def candidate_score(retarget, chain, q, final_target, wrist_base, elbow_base, shoulder_base, reference_q, config):
    transform = chain.fk(q)[0]
    tcp = transform[:3, 3]
    position_error = float(np.linalg.norm(np.asarray(final_target, dtype=float) - tcp))
    metrics = retarget.compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
    position_score = position_error / max(float(config["position_tolerance"]), 1e-6)
    score = (
        float(config["position_score_weight"]) * position_score
        + float(config["forearm_score_weight"]) * float(metrics.get("forearm_max_rad", 0.0))
        + float(config["upper_arm_score_weight"]) * float(metrics.get("upper_arm_max_rad", 0.0))
        + float(config["included_angle_score_weight"]) * float(metrics.get("included_angle_abs_error_rad", 0.0))
        + float(config["joint_update_score_weight"]) * float(np.linalg.norm(q - reference_q))
        + float(config["anti_self_score_weight"]) * float(metrics.get("anti_self_penalty", 0.0))
    )
    tcp_z = float(tcp[2])
    joint_margin = finite_joint_margin(q, chain.limits)
    safe = tcp_z >= float(config["min_tcp_z_base"]) and joint_margin >= float(config["min_joint_margin_rad"])
    return score, position_error, metrics, safe, tcp_z, joint_margin


def mirror_parallel_candidate_step(
    retarget,
    chain,
    q,
    target,
    wrist_base,
    elbow_base,
    shoulder_base,
    config,
):
    start_time = time.perf_counter()
    seed_score, seed_error, seed_metrics, _, _, _ = candidate_score(
        retarget,
        chain,
        q,
        target,
        wrist_base,
        elbow_base,
        shoulder_base,
        q,
        config,
    )
    current_tcp = chain.fk(q)[0][:3, 3]
    def solve_candidate(alpha):
        alpha = float(alpha)
        continuation_target = current_tcp + alpha * (np.asarray(target, dtype=float) - current_tcp)
        cq, cconverged, cinner_error, _, _, _ = chain.solve_functional_hierarchical(
            continuation_target,
            wrist_base,
            elbow_base,
            shoulder_base,
            q,
            robot_shoulder_anchor=retarget.robot_shoulder_anchor_base(),
            elbow_joint_index=retarget.robot_elbow_joint_index(),
            shoulder_joint_index=retarget.robot_shoulder_joint_index(),
            iterations=int(config["candidate_iterations"]),
            primary_damping=float(config["primary_damping"]),
            secondary_damping=float(config["secondary_damping"]),
            position_tolerance=float(config["position_tolerance"]),
            forearm_tolerance=math.radians(float(config["forearm_tolerance_deg"])),
            upper_arm_tolerance=math.radians(float(config["upper_arm_tolerance_deg"])),
            forearm_weight=float(config["forearm_weight"]),
            upper_arm_weight=float(config["upper_arm_weight"]),
            included_angle_weight=float(config["included_angle_weight"]),
            secondary_step_scale=float(config["secondary_step_scale"]),
            max_step=float(config["max_step"]),
            finite_difference_step=float(config["finite_difference_step"]),
            secondary_position_guard=float(config["secondary_position_guard"]),
        )
        score, final_error, metrics, safe, tcp_z, joint_margin = candidate_score(
            retarget,
            chain,
            cq,
            target,
            wrist_base,
            elbow_base,
            shoulder_base,
            q,
            config,
        )
        progress = float(seed_score - score)
        accepted = bool(safe and progress >= float(config["lyapunov_eta"]))
        return {
            "alpha": alpha,
            "q": cq,
            "score": score,
            "progress": progress,
            "accepted": accepted,
            "inner_converged": bool(cconverged),
            "inner_position_error_m": float(cinner_error),
            "final_position_error_m": final_error,
            "metrics": metrics,
            "safe": safe,
            "tcp_z_base": tcp_z,
            "joint_margin_rad": joint_margin,
        }

    alphas = list(config["alphas"])
    worker_count = max(1, int(config.get("parallel_workers", 1)))
    if worker_count > 1 and len(alphas) > 1:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(alphas))) as executor:
            candidates = list(executor.map(solve_candidate, alphas))
    else:
        candidates = [solve_candidate(alpha) for alpha in alphas]
    accepted_candidates = [item for item in candidates if item["accepted"]]
    if accepted_candidates:
        selected = max(accepted_candidates, key=lambda item: (item["alpha"], item["progress"]))
        selection_reason = "largest_accepted_alpha"
    else:
        selected = min(candidates, key=lambda item: (not item["safe"], item["score"]))
        selection_reason = "best_score_fallback"
    final_metrics = dict(selected["metrics"])
    final_metrics.update(
        {
            "mirror_selected_alpha": float(selected["alpha"]),
            "mirror_selection_reason": selection_reason,
            "mirror_seed_score": float(seed_score),
            "mirror_selected_score": float(selected["score"]),
            "mirror_progress": float(selected["progress"]),
            "mirror_accepted_candidates": int(len(accepted_candidates)),
            "mirror_candidate_count": int(len(candidates)),
            "mirror_solve_time_ms": float(1000.0 * (time.perf_counter() - start_time)),
            "mirror_final_position_error_m": float(selected["final_position_error_m"]),
            "mirror_inner_position_error_m": float(selected["inner_position_error_m"]),
            "mirror_tcp_z_base": float(selected["tcp_z_base"]),
            "mirror_joint_margin_rad": float(selected["joint_margin_rad"]),
        }
    )
    converged = selected["final_position_error_m"] < float(config["position_tolerance"])
    return (
        selected["q"],
        converged,
        float(selected["final_position_error_m"]),
        float(final_metrics.get("forearm_max_rad", 0.0)),
        float(final_metrics.get("upper_arm_max_rad", 0.0)),
        final_metrics,
    )


def build_mirror_trajectory(retarget, chain, seed, trajectory, seconds, fps, mirror_config):
    frame_count = int(round(seconds * fps))
    q = np.asarray(seed, dtype=float).copy()
    start_tcp = chain.fk(q)[0][:3, 3]
    ik_config = retarget.RETARGET_RUNTIME_CONFIG.get("ik", {})
    controller_blend_in = float(ik_config.get("controller_blend_in_seconds", 3.0))
    phase_prefix = retarget.loaded_trajectory_phase_prefix()
    points = []
    diagnostics = []
    for frame in range(frame_count + 1):
        t = min(seconds, frame / fps)
        point_world, point_base, observed_phase, arm = retarget.sample_reconstructed_target(trajectory, t)
        raw_base_target = (
            retarget.world_to_robot_base(point_world)
            if point_base is None
            else np.asarray(point_base, dtype=float)
        )
        observed_target = retarget.clip_base_target(retarget.conditioned_base_target(raw_base_target, t))
        target = observed_target
        if t < controller_blend_in:
            alpha = smoothstep(t / max(0.1, controller_blend_in))
            target = start_tcp * (1.0 - alpha) + observed_target * alpha
        phase = f"{phase_prefix}-{observed_phase}"
        wrist_base = observed_target
        elbow_base = retarget.global_workspace_transform_base(retarget.world_to_robot_base(arm["right_elbow_world"]))
        shoulder_base = retarget.global_workspace_transform_base(retarget.world_to_robot_base(arm["right_shoulder_world"]))
        q_before = q.copy()
        if frame == 0:
            transform = chain.fk(q)[0]
            error = float(np.linalg.norm(np.asarray(target, dtype=float) - transform[:3, 3]))
            functional_metrics = retarget.compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
            converged = error < float(mirror_config["position_tolerance"])
        elif t < controller_blend_in:
            q, converged, error = chain.solve_position(target, q)
            transform = chain.fk(q)[0]
            functional_metrics = retarget.compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
        else:
            q, converged, error, _, _, functional_metrics = mirror_parallel_candidate_step(
                retarget,
                chain,
                q,
                target,
                wrist_base,
                elbow_base,
                shoulder_base,
                mirror_config,
            )
            transform = chain.fk(q)[0]
        diagnostics.append(
            {
                "time": round(t, 4),
                "phase": phase,
                "target": np.asarray(target, dtype=float).tolist(),
                "tcp": transform[:3, 3].tolist(),
                "error": float(error),
                "orientation_error": float(functional_metrics.get("forearm_max_rad", 0.0)),
                "elbow_error": float(functional_metrics.get("upper_arm_max_rad", 0.0)),
                "functional_metrics": jsonable(functional_metrics),
                "joint_delta_limited": False,
                "converged": bool(converged),
                "mirror_joint_delta_norm_rad": float(np.linalg.norm(q - q_before)),
            }
        )
        points.append(q.copy())
    return np.asarray(points), diagnostics


def main():
    parser = argparse.ArgumentParser(description="MIRROR-style parallel candidate IK baseline for xArm7 retargeting.")
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--retarget-config", required=True)
    parser.add_argument("--mirror-config")
    parser.add_argument("--seed-joint-trajectory", help="Optional solved trajectory whose first point seeds this run.")
    parser.add_argument("--retarget-script", default="/tmp/retarget-smplx-hand-to-xarm.py")
    parser.add_argument("--urdf-text")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    retarget = load_retarget_module(args.retarget_script)
    retarget_config = load_json(args.retarget_config)
    retarget.apply_retarget_config(retarget_config)
    mirror_config = deep_merge(DEFAULT_MIRROR_CONFIG, load_json(args.mirror_config, {}))["mirror"]

    urdf_text = get_urdf_text(retarget, args.urdf_text)
    chain = retarget.KinematicChain(urdf_text, base_link=retarget.BASE_LINK, tip_link=retarget.TIP_LINK)
    trajectory = retarget.load_reconstructed_trajectory(args.trajectory_json)
    if args.seed_joint_trajectory:
        seed_payload = load_json(args.seed_joint_trajectory)
        seed = np.asarray(seed_payload["positions"][0], dtype=float)
    else:
        seed = np.zeros(len(retarget.JOINT_NAMES), dtype=float)

    start = time.perf_counter()
    positions, diagnostics = build_mirror_trajectory(
        retarget,
        chain,
        seed,
        trajectory,
        float(args.seconds),
        float(args.fps),
        mirror_config,
    )
    elapsed = time.perf_counter() - start
    mirror_items = [
        item["functional_metrics"]
        for item in diagnostics
        if item.get("functional_metrics") and "mirror_selected_alpha" in item["functional_metrics"]
    ]
    payload = {
        "schema": "xarm7_joint_trajectory/v1",
        "source_trajectory": args.trajectory_json,
        "retarget_config": args.retarget_config,
        "mirror_config": args.mirror_config or "builtin_defaults",
        "seconds": float(args.seconds),
        "fps": float(args.fps),
        "joint_names": retarget.JOINT_NAMES,
        "positions": [[float(value) for value in row] for row in positions],
        "diagnostics": jsonable(diagnostics),
        "mirror_runtime": {
            "seconds": float(elapsed),
            "frames": int(len(positions)),
            "mean_ms_per_frame": float(1000.0 * elapsed / max(1, len(positions))),
            "mean_candidate_solve_ms": float(
                np.mean([item["mirror_solve_time_ms"] for item in mirror_items]) if mirror_items else 0.0
            ),
            "mean_selected_alpha": float(
                np.mean([item["mirror_selected_alpha"] for item in mirror_items]) if mirror_items else 0.0
            ),
            "mean_accepted_candidates": float(
                np.mean([item["mirror_accepted_candidates"] for item in mirror_items]) if mirror_items else 0.0
            ),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["mirror_runtime"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
