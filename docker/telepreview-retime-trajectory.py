#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


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


def interpolate_vectors(a, b, alpha):
    return (1.0 - alpha) * np.asarray(a, dtype=float) + alpha * np.asarray(b, dtype=float)


def sample_source_diagnostic(diagnostics, source_time, source_fps):
    if not diagnostics:
        return {}, 0, 0.0
    raw_index = source_time * source_fps
    lo = int(math.floor(raw_index))
    hi = min(len(diagnostics) - 1, lo + 1)
    lo = max(0, min(len(diagnostics) - 1, lo))
    alpha = max(0.0, min(1.0, raw_index - lo))
    first = diagnostics[lo]
    second = diagnostics[hi]
    nearest = first if alpha < 0.5 else second
    target = np.asarray(nearest.get("target", [0.0, 0.0, 0.0]), dtype=float)
    if first.get("target") is not None and second.get("target") is not None:
        target = interpolate_vectors(first["target"], second["target"], alpha)
    return nearest, lo, target


def soft_limit_value(value, lower, upper, margin):
    if math.isfinite(lower) and value < lower + margin:
        value = lower + margin
    if math.isfinite(upper) and value > upper - margin:
        value = upper - margin
    return value


def choose_equivalent_angle(value, previous, lower, upper, margin):
    candidates = []
    for turn in range(-2, 3):
        candidate = value + (2.0 * math.pi * turn)
        if math.isfinite(lower) and candidate < lower:
            continue
        if math.isfinite(upper) and candidate > upper:
            continue
        soft_candidate = soft_limit_value(candidate, lower, upper, margin)
        candidates.append(soft_candidate)
    if not candidates:
        return soft_limit_value(value, lower, upper, margin)
    return min(candidates, key=lambda item: abs(item - previous))


def smooth_equivalent_angles(positions, limits, margin):
    result = np.asarray(positions, dtype=float).copy()
    if len(result) == 0:
        return result
    for joint_idx, (lower, upper) in enumerate(limits):
        span = upper - lower if math.isfinite(lower) and math.isfinite(upper) else math.inf
        result[0, joint_idx] = soft_limit_value(result[0, joint_idx], lower, upper, margin)
        for row_idx in range(1, len(result)):
            if span >= (2.0 * math.pi - 1e-3):
                result[row_idx, joint_idx] = choose_equivalent_angle(
                    result[row_idx, joint_idx],
                    result[row_idx - 1, joint_idx],
                    lower,
                    upper,
                    margin,
                )
            else:
                result[row_idx, joint_idx] = soft_limit_value(result[row_idx, joint_idx], lower, upper, margin)
    return result


def add_retimed_segment(out_positions, out_source_times, out_keyframes, start_q, end_q, start_t, end_t, max_step):
    delta = np.asarray(end_q, dtype=float) - np.asarray(start_q, dtype=float)
    steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / max(max_step, 1e-9))))
    for step in range(1, steps + 1):
        alpha = step / steps
        out_positions.append((1.0 - alpha) * start_q + alpha * end_q)
        out_source_times.append((1.0 - alpha) * start_t + alpha * end_t)
        out_keyframes.append(step == steps)


def retime_positions(positions, source_fps, max_step, prepose_hold_seconds):
    retimed = [np.asarray(positions[0], dtype=float)]
    source_times = [0.0]
    keyframes = [True]
    hold_frames = int(round(max(0.0, prepose_hold_seconds) * source_fps))
    for _ in range(hold_frames):
        retimed.append(retimed[0].copy())
        source_times.append(0.0)
        keyframes.append(False)
    for idx in range(len(positions) - 1):
        add_retimed_segment(
            retimed,
            source_times,
            keyframes,
            positions[idx],
            positions[idx + 1],
            idx / source_fps,
            (idx + 1) / source_fps,
            max_step,
        )
    return np.asarray(retimed, dtype=float), np.asarray(source_times, dtype=float), keyframes


def build_diagnostics(chain, retimed_positions, source_times, keyframes, source_diagnostics, source_fps):
    diagnostics = []
    for frame_idx, (q, source_time) in enumerate(zip(retimed_positions, source_times)):
        nearest, _, target = sample_source_diagnostic(source_diagnostics, float(source_time), source_fps)
        transform, _, _ = chain.fk(q)
        tcp = transform[:3, 3]
        phase = nearest.get("phase", "telepreview-retimed")
        if not keyframes[frame_idx]:
            target = tcp.copy()
            phase = "telepreview-retime-transition"
        diagnostic = {
            "time": round(frame_idx / source_fps, 4),
            "source_time": round(float(source_time), 4),
            "phase": phase,
            "target": target.tolist(),
            "tcp": tcp.tolist(),
            "error": float(np.linalg.norm(tcp - target)),
            "orientation_error": nearest.get("orientation_error"),
            "elbow_error": nearest.get("elbow_error"),
            "functional_metrics": nearest.get("functional_metrics"),
            "joint_delta_limited": False,
            "converged": bool(nearest.get("converged", True)),
            "telepreview_retimed": True,
            "telepreview_source_keyframe": bool(keyframes[frame_idx]),
        }
        if frame_idx == 0:
            diagnostic["phase"] = "telepreview-prepose"
        diagnostics.append(diagnostic)
    return diagnostics


def trajectory_stats(positions, fps):
    if len(positions) < 2:
        return {
            "joint_step_abs_max_rad": 0.0,
            "joint_step_p95_rad": 0.0,
            "joint_velocity_p95_rad_s": 0.0,
        }
    steps = np.max(np.abs(np.diff(positions, axis=0)), axis=1)
    return {
        "joint_step_abs_max_rad": float(np.max(steps)),
        "joint_step_p95_rad": float(np.percentile(steps, 95)),
        "joint_velocity_p95_rad_s": float(np.percentile(steps * fps, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Apply TelePreview prepose + retiming to a solved xArm trajectory.")
    parser.add_argument("--xarm-json", required=True)
    parser.add_argument("--retarget-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retarget-script", default="/tmp/retarget-smplx-hand-to-xarm.py")
    parser.add_argument("--urdf-text")
    parser.add_argument("--max-step-rad", type=float, default=0.45)
    parser.add_argument("--soft-limit-margin-rad", type=float, default=0.015)
    parser.add_argument("--prepose-hold-seconds", type=float, default=0.5)
    args = parser.parse_args()

    payload = load_json(args.xarm_json)
    retarget_config = load_json(args.retarget_config)
    positions = np.asarray(payload["positions"], dtype=float)
    source_fps = float(payload.get("fps", 10.0))
    if len(positions) == 0:
        raise RuntimeError("input trajectory has no positions")

    retarget = load_retarget_module(args.retarget_script)
    retarget.apply_retarget_config(retarget_config)
    urdf_text = get_urdf_text(retarget, args.urdf_text)
    chain = retarget.KinematicChain(urdf_text, base_link=retarget.BASE_LINK, tip_link=retarget.TIP_LINK)

    source_stats = trajectory_stats(positions, source_fps)
    softened_positions = smooth_equivalent_angles(positions, chain.limits, float(args.soft_limit_margin_rad))
    retimed_positions, source_times, keyframes = retime_positions(
        softened_positions,
        source_fps,
        float(args.max_step_rad),
        float(args.prepose_hold_seconds),
    )
    diagnostics = build_diagnostics(
        chain,
        retimed_positions,
        source_times,
        keyframes,
        payload.get("diagnostics", []),
        source_fps,
    )
    retimed_stats = trajectory_stats(retimed_positions, source_fps)
    output = {
        "schema": "xarm7_joint_trajectory/v1",
        "source_trajectory": payload.get("source_trajectory"),
        "retarget_config": payload.get("retarget_config"),
        "seconds": float((len(retimed_positions) - 1) / source_fps),
        "fps": source_fps,
        "joint_names": payload.get("joint_names"),
        "positions": [[float(value) for value in row] for row in retimed_positions],
        "diagnostics": jsonable(diagnostics),
        "telepreview_retiming": {
            "schema": "telepreview_prepose_retime/v1",
            "input_xarm_json": args.xarm_json,
            "prepose_hold_seconds": float(args.prepose_hold_seconds),
            "max_step_rad": float(args.max_step_rad),
            "soft_limit_margin_rad": float(args.soft_limit_margin_rad),
            "source_points": int(len(positions)),
            "retimed_points": int(len(retimed_positions)),
            "source_keyframes": int(sum(1 for value in keyframes if value)),
            "transition_points": int(sum(1 for value in keyframes if not value)),
            "source_seconds": float(payload.get("seconds", (len(positions) - 1) / source_fps)),
            "retimed_seconds": float((len(retimed_positions) - 1) / source_fps),
            "source_stats": source_stats,
            "retimed_stats": retimed_stats,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["telepreview_retiming"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
