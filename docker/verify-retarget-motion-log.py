#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path


SEGMENT_RE = re.compile(
    r"^segment\s+(?P<name>\S+)\s+to=(?P<waypoint>\S+)\s+result=(?P<result>-?\d+)\s+"
    r"duration=(?P<duration>[0-9.]+)\s+target=\[(?P<target>[^\]]+)\]\s+"
    r"tcp=\[(?P<tcp>[^\]]+)\]\s+err=(?P<err>[0-9.]+)"
)


def parse_vec(text):
    return [float(part) for part in text.split()]


def xy_distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def main():
    parser = argparse.ArgumentParser(description="Verify segmented xArm7 retarget motion completeness from motion.log.")
    parser.add_argument("log_path")
    parser.add_argument("--min-approach-xy", type=float, default=0.20)
    parser.add_argument("--min-descend-z", type=float, default=0.09)
    parser.add_argument("--min-lift-z", type=float, default=0.25)
    parser.add_argument("--max-ready-error", type=float, default=0.08)
    parser.add_argument("--max-hover-error", type=float, default=0.06)
    parser.add_argument("--max-grasp-error", type=float, default=0.025)
    parser.add_argument("--max-lift-error", type=float, default=0.10)
    args = parser.parse_args()

    segments = {}
    trajectory_result = None
    for line in Path(args.log_path).read_text(encoding="utf-8").splitlines():
        match = SEGMENT_RE.match(line.strip())
        if match:
            segments[match.group("name")] = {
                "waypoint": match.group("waypoint"),
                "result": int(match.group("result")),
                "duration": float(match.group("duration")),
                "target": parse_vec(match.group("target")),
                "tcp": parse_vec(match.group("tcp")),
                "err": float(match.group("err")),
            }
        if line.startswith("trajectory_result="):
            trajectory_result = int(line.split("=", 1)[1])

    required = [
        "settle-ready",
        "approach-hover",
        "hover-hold",
        "descend-grasp",
        "hold-grasp",
        "vertical-lift",
        "lift-hold",
    ]
    missing = [name for name in required if name not in segments]
    failures = []
    if missing:
        failures.append(f"missing_segments={','.join(missing)}")

    if not missing:
        ready = segments["settle-ready"]["tcp"]
        hover = segments["approach-hover"]["tcp"]
        grasp = segments["descend-grasp"]["tcp"]
        hold = segments["hold-grasp"]["tcp"]
        lift = segments["vertical-lift"]["tcp"]

        approach_xy = xy_distance(ready, hover)
        descend_z = hover[2] - grasp[2]
        lift_z = lift[2] - hold[2]

        print(f"approach_xy={approach_xy:.4f}m")
        print(f"descend_z={descend_z:.4f}m")
        print(f"lift_z={lift_z:.4f}m")
        print(f"ready_err={segments['settle-ready']['err']:.4f}m")
        print(f"hover_err={segments['approach-hover']['err']:.4f}m")
        print(f"grasp_err={segments['descend-grasp']['err']:.4f}m")
        print(f"hold_err={segments['hold-grasp']['err']:.4f}m")
        print(f"lift_err={segments['vertical-lift']['err']:.4f}m")
        print(f"trajectory_result={trajectory_result}")

        if approach_xy < args.min_approach_xy:
            failures.append(f"approach_xy<{args.min_approach_xy:.3f}")
        if descend_z < args.min_descend_z:
            failures.append(f"descend_z<{args.min_descend_z:.3f}")
        if lift_z < args.min_lift_z:
            failures.append(f"lift_z<{args.min_lift_z:.3f}")
        if segments["settle-ready"]["err"] > args.max_ready_error:
            failures.append(f"ready_err>{args.max_ready_error:.3f}")
        if segments["approach-hover"]["err"] > args.max_hover_error:
            failures.append(f"hover_err>{args.max_hover_error:.3f}")
        if segments["descend-grasp"]["err"] > args.max_grasp_error:
            failures.append(f"grasp_err>{args.max_grasp_error:.3f}")
        if segments["hold-grasp"]["err"] > args.max_grasp_error:
            failures.append(f"hold_err>{args.max_grasp_error:.3f}")
        if segments["vertical-lift"]["err"] > args.max_lift_error:
            failures.append(f"lift_err>{args.max_lift_error:.3f}")
        if trajectory_result != 0:
            failures.append("trajectory_result!=0")

    if failures:
        print("motion_verification=FAIL")
        for failure in failures:
            print(f"failure={failure}")
        return 1
    print("motion_verification=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
