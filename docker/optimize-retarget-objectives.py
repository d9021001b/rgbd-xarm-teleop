#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import copy
from pathlib import Path

import numpy as np


def load_retarget_module(path):
    spec = importlib.util.spec_from_file_location("retarget_smplx_xarm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def clean_urdf(retarget, path):
    return retarget.clean_robot_description(Path(path).read_text(encoding="utf-8").strip())


def summarize_diagnostics(diagnostics):
    pos = np.asarray([item["error"] for item in diagnostics], dtype=float)
    functional = [item["functional_metrics"] for item in diagnostics if item.get("functional_metrics")]
    forearm_max = np.asarray([item["forearm_max_rad"] for item in functional], dtype=float)
    upper_max = np.asarray([item["upper_arm_max_rad"] for item in functional], dtype=float)
    included = np.asarray([item.get("included_angle_abs_error_rad", 0.0) for item in functional], dtype=float)
    orient = np.asarray([item.get("orientation_norm_rad", 0.0) for item in functional], dtype=float)
    anti = np.asarray([item.get("anti_self_penalty", 0.0) for item in functional], dtype=float)
    return {
        "position_mean_m": float(np.mean(pos)),
        "position_max_m": float(np.max(pos)),
        "forearm_mean_deg": float(np.degrees(np.mean(forearm_max))) if len(forearm_max) else None,
        "forearm_max_deg": float(np.degrees(np.max(forearm_max))) if len(forearm_max) else None,
        "upper_mean_deg": float(np.degrees(np.mean(upper_max))) if len(upper_max) else None,
        "upper_max_deg": float(np.degrees(np.max(upper_max))) if len(upper_max) else None,
        "included_mean_deg": float(np.degrees(np.mean(included))) if len(included) else None,
        "included_max_deg": float(np.degrees(np.max(included))) if len(included) else None,
        "orientation_mean_deg": float(np.degrees(np.mean(orient))) if len(orient) else None,
        "anti_max": float(np.max(anti)) if len(anti) else 0.0,
        "converged_count": int(sum(1 for item in diagnostics if item.get("converged"))),
        "joint_delta_limited_count": int(sum(1 for item in diagnostics if item.get("joint_delta_limited"))),
    }


def score_summary(summary):
    pos_mean = summary["position_mean_m"]
    pos_max = summary["position_max_m"]
    forearm = summary["forearm_mean_deg"] or 180.0
    upper = summary["upper_mean_deg"] or 180.0
    forearm_max = summary["forearm_max_deg"] or 180.0
    upper_max = summary["upper_max_deg"] or 180.0
    included = summary["included_mean_deg"] or 180.0
    included_max = summary["included_max_deg"] or 180.0
    score = forearm + 1.15 * upper + 0.12 * forearm_max + 0.12 * upper_max
    score += 0.85 * included + 0.08 * included_max
    score += 2500.0 * max(0.0, pos_mean - 0.004)
    score += 6500.0 * max(0.0, pos_max - 0.020)
    score += 4.0 * summary["anti_max"]
    return float(score)


def candidate_updates(family="all"):
    candidates = []
    # Keep the current table-edge strategy, but progressively give the null-space
    # arm objectives more authority and remove joint regularization that freezes
    # elbow swivel.
    if family in ("all", "table"):
      best_rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
      for name, forearm_weight, upper_weight, guard, step_scale, joint_reg in [
        ("table_rotbest_fw0.08_uw0.04", 0.08, 0.04, 0.045, 0.45, 0.08),
        ("table_rotbest_fw0.35_uw0.20", 0.35, 0.20, 0.070, 0.70, 0.02),
        ("table_rotbest_fw0.60_uw0.35", 0.60, 0.35, 0.090, 0.85, 0.01),
        ("table_rotbest_fw1.2_uw0.7", 1.20, 0.70, 0.120, 1.00, 0.0),
      ]:
        candidates.append(
          {
            "name": name,
            "updates": {
                "orientation": {"mode": "table_edge_soft_functional", "downward_blend": 0.10},
                "functional": {
                    "source_vector_rotation_base": best_rotation,
                    "forearm_weight": forearm_weight,
                    "upper_arm_weight": upper_weight,
                    "joint_regularization_weight": joint_reg,
                    "secondary_step_scale": step_scale,
                    "secondary_position_guard": guard,
                    "max_step": 0.080,
                },
                "ik": {
                    "max_iterations": 280,
                    "orientation_weight": 0.035,
                    "orientation_tolerance": 0.80,
                    "position_tolerance": 0.014,
                },
                "anti_self_insertion": {"weight": 0.20, "min_tool_alignment": 0.05},
            },
          }
        )
      for included_weight in [0.04, 0.08, 0.12, 0.20, 0.32]:
        candidates.append(
          {
            "name": f"table_rotbest_inc{included_weight:.2f}_fw0.08_uw0.04",
            "updates": {
                "orientation": {"mode": "table_edge_soft_functional", "downward_blend": 0.10},
                "functional": {
                    "source_vector_rotation_base": best_rotation,
                    "forearm_weight": 0.08,
                    "upper_arm_weight": 0.04,
                    "included_angle_weight": included_weight,
                    "joint_regularization_weight": 0.08,
                    "secondary_step_scale": 0.45,
                    "secondary_position_guard": 0.045,
                    "max_step": 0.080,
                },
                "ik": {
                    "max_iterations": 280,
                    "orientation_weight": 0.035,
                    "orientation_tolerance": 0.80,
                    "position_tolerance": 0.014,
                },
                "anti_self_insertion": {"weight": 0.20, "min_tool_alignment": 0.05},
            },
          }
        )
      for included_weight in [0.08, 0.16]:
        candidates.append(
          {
            "name": f"table_rotbest_inc{included_weight:.2f}_fw0.16_uw0.08",
            "updates": {
                "orientation": {"mode": "table_edge_soft_functional", "downward_blend": 0.10},
                "functional": {
                    "source_vector_rotation_base": best_rotation,
                    "forearm_weight": 0.16,
                    "upper_arm_weight": 0.08,
                    "included_angle_weight": included_weight,
                    "joint_regularization_weight": 0.06,
                    "secondary_step_scale": 0.50,
                    "secondary_position_guard": 0.052,
                    "max_step": 0.080,
                },
                "ik": {
                    "max_iterations": 280,
                    "orientation_weight": 0.035,
                    "orientation_tolerance": 0.80,
                    "position_tolerance": 0.014,
                },
                "anti_self_insertion": {"weight": 0.20, "min_tool_alignment": 0.05},
            },
          }
        )
      for forearm_weight, upper_weight in [(0.70, 0.30), (1.20, 0.70), (2.00, 1.20)]:
        candidates.append(
            {
                "name": f"table_fw{forearm_weight}_uw{upper_weight}_loose",
                "updates": {
                    "orientation": {"mode": "table_edge_soft_functional", "downward_blend": 0.10},
                    "functional": {
                        "forearm_weight": forearm_weight,
                        "upper_arm_weight": upper_weight,
                        "joint_regularization_weight": 0.0,
                        "secondary_step_scale": 1.0,
                        "secondary_position_guard": 0.12,
                        "max_step": 0.080,
                    },
                    "ik": {
                        "max_iterations": 260,
                        "orientation_weight": 0.035,
                        "orientation_tolerance": 0.80,
                        "position_tolerance": 0.014,
                    },
                    "anti_self_insertion": {"weight": 0.20, "min_tool_alignment": 0.05},
                },
            }
        )

    # Remove the table-edge gripper orientation and let the arm shape define the
    # null-space objective. This is the most direct teleoperation consistency test.
    if family in ("all", "functional"):
      best_rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
      candidates.append(
        {
            "name": "functional_rotbest_fw1.4_uw0.9",
            "updates": {
                "orientation": {"mode": "functional_hierarchical"},
                "functional": {
                    "source_vector_rotation_base": best_rotation,
                    "forearm_weight": 1.40,
                    "upper_arm_weight": 0.90,
                    "secondary_step_scale": 1.0,
                    "secondary_position_guard": 0.16,
                    "max_step": 0.085,
                    "joint_regularization_weight": 0.0,
                    "forearm_tolerance_deg": 35.0,
                    "upper_arm_tolerance_deg": 45.0,
                    "branch_recovery_enabled": False,
                },
                "ik": {
                    "max_iterations": 220,
                    "position_tolerance": 0.014,
                    "max_frame_joint_delta_rad": 10.0,
                },
                "anti_self_insertion": {"weight": 0.0},
            },
        }
      )
      for forearm_weight, upper_weight in [(1.40, 0.90), (2.20, 1.60), (3.00, 2.40), (4.00, 3.20)]:
        candidates.append(
            {
                "name": f"functional_fw{forearm_weight}_uw{upper_weight}",
                "updates": {
                    "orientation": {"mode": "functional_hierarchical"},
                    "functional": {
                        "forearm_weight": forearm_weight,
                        "upper_arm_weight": upper_weight,
                        "secondary_step_scale": 1.0,
                        "secondary_position_guard": 0.16,
                        "max_step": 0.085,
                        "joint_regularization_weight": 0.0,
                        "forearm_tolerance_deg": 35.0,
                        "upper_arm_tolerance_deg": 45.0,
                        "branch_recovery_enabled": False,
                        "branch_recovery_offsets_rad": [-2.4, -1.4, -0.7, 0.7, 1.4, 2.4],
                        "branch_recovery_forearm_weight": 1.7,
                        "branch_recovery_upper_arm_weight": 1.5,
                        "branch_recovery_joint_weight": 0.025,
                    },
                    "ik": {
                        "max_iterations": 220,
                        "position_tolerance": 0.014,
                        "max_frame_joint_delta_rad": 10.0,
                    },
                    "anti_self_insertion": {"weight": 0.0},
                },
            }
        )
    return candidates


def solve_candidate(retarget, chain, reconstructed, seed_q, base_config, candidate, seconds, fps):
    config = deep_update(copy.deepcopy(base_config), candidate["updates"])
    retarget.apply_retarget_config(config)
    positions, diagnostics = retarget.build_trajectory(
        chain,
        seed_q,
        seconds,
        fps,
        reconstructed,
        cup_constrained=False,
    )
    summary = summarize_diagnostics(diagnostics)
    summary["score"] = score_summary(summary)
    return config, positions, diagnostics, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--seed-trajectory", required=True)
    parser.add_argument("--retarget-script", default="/tmp/retarget-smplx-hand-to-xarm.py")
    parser.add_argument("--urdf-text", default="/tmp/xarm_robot_description_param.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--family", choices=["all", "table", "functional"], default="all")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--only-name", help="Run just one candidate by exact name.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retarget = load_retarget_module(args.retarget_script)
    base_config = retarget.load_retarget_config(args.base_config)
    retarget.apply_retarget_config(base_config)
    chain = retarget.KinematicChain(clean_urdf(retarget, args.urdf_text), base_link=retarget.BASE_LINK, tip_link=retarget.TIP_LINK)
    reconstructed = retarget.load_reconstructed_trajectory(args.trajectory_json)
    seed_payload = json.loads(Path(args.seed_trajectory).read_text(encoding="utf-8"))
    seed_q = np.asarray(seed_payload["positions"][0], dtype=float)

    baseline = {
        "name": "baseline",
        "updates": {},
    }
    results = []
    best = None
    candidates = candidate_updates(args.family)
    if not args.skip_baseline:
        candidates = [baseline] + candidates
    if args.only_name:
        candidates = [candidate for candidate in candidates if candidate["name"] == args.only_name]
        if not candidates:
            raise RuntimeError(f"No candidate matched --only-name {args.only_name!r}")
    for idx, candidate in enumerate(candidates, start=1):
        config, positions, diagnostics, summary = solve_candidate(
            retarget,
            chain,
            reconstructed,
            seed_q,
            base_config,
            candidate,
            args.seconds,
            args.fps,
        )
        summary["name"] = candidate["name"]
        summary["candidate_index"] = idx
        results.append(summary)
        print(
            f"{idx:02d} {candidate['name']:<32} "
            f"score={summary['score']:.2f} pos_mean={summary['position_mean_m']:.4f} "
            f"pos_max={summary['position_max_m']:.4f} "
            f"forearm_mean={summary['forearm_mean_deg']:.1f} upper_mean={summary['upper_mean_deg']:.1f} "
            f"included_mean={summary['included_mean_deg']:.1f} "
            f"forearm_max={summary['forearm_max_deg']:.1f} upper_max={summary['upper_max_deg']:.1f} "
            f"included_max={summary['included_max_deg']:.1f}",
            flush=True,
        )
        if best is None or summary["score"] < best["summary"]["score"]:
            best = {
                "candidate": candidate,
                "config": config,
                "positions": positions,
                "diagnostics": diagnostics,
                "summary": summary,
            }

    results.sort(key=lambda item: item["score"])
    (out_dir / "objective_sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    best_config = best["config"]
    best_config.setdefault("execution", {})
    best_config["execution"]["time_scale"] = 0.6
    best_config["execution"]["path_tolerance_rad"] = 100.0
    best_config["execution"]["goal_tolerance_rad"] = 100.0
    best_config_path = out_dir / "xarm7_smplx_retarget_optimized_objectives.json"
    best_config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    trajectory_payload = {
        "schema": "xarm7_joint_trajectory/v1",
        "source_trajectory": args.trajectory_json,
        "retarget_config": str(best_config_path),
        "seconds": float(args.seconds),
        "fps": float(args.fps),
        "joint_names": retarget.JOINT_NAMES,
        "positions": [[float(v) for v in row] for row in best["positions"]],
        "diagnostics": retarget.jsonable(best["diagnostics"]),
    }
    best_trajectory_path = out_dir / "xarm7_smplx_retarget_optimized_objectives_trajectory.json"
    best_trajectory_path.write_text(json.dumps(trajectory_payload, indent=2), encoding="utf-8")

    print("BEST", json.dumps(best["summary"], indent=2))
    print(f"best_config={best_config_path}")
    print(f"best_trajectory={best_trajectory_path}")


if __name__ == "__main__":
    raise SystemExit(main())
