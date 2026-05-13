#!/usr/bin/env python3
import argparse
import math
import subprocess
import time


def quat_from_z_axis_to_vector(v):
    x, y, z = v
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z = x / norm, y / norm, z / norm
    ax, ay, az = -y, x, 0.0
    dot = max(-1.0, min(1.0, z))
    angle = math.acos(dot)
    axis_norm = math.sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0 if dot > 0 else 0.0)
    ax, ay, az = ax / axis_norm, ay / axis_norm, az / axis_norm
    half = angle / 2.0
    s = math.sin(half)
    return (ax * s, ay * s, az * s, math.cos(half))


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def mul(v, scalar):
    return tuple(v[i] * scalar for i in range(3))


def set_pose(name, position, quat):
    x, y, z = position
    qx, qy, qz, qw = quat
    req = (
        f'name: "{name}" '
        f'position: {{x: {x:.4f} y: {y:.4f} z: {z:.4f}}} '
        f'orientation: {{x: {qx:.6f} y: {qy:.6f} z: {qz:.6f} w: {qw:.6f}}}'
    )
    subprocess.run(
        [
            "gz", "service",
            "-s", "/world/default/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", req,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def bent_elbow(shoulder, hand):
    reach = sub(hand, shoulder)
    base = add(shoulder, mul(reach, 0.48))
    extension = min(1.0, math.sqrt(sum(axis * axis for axis in reach)) / 0.95)
    bend = (0.02, -0.18 * (1.0 - 0.35 * extension), -0.16 * (1.0 - 0.25 * extension))
    return add(base, bend)


def move_segment(name, shoulder, source, target, seconds, hz):
    steps = max(2, int(seconds * hz))
    for i in range(steps + 1):
        t = smoothstep(i / steps)
        hand = lerp(source, target, t)
        elbow = bent_elbow(shoulder, hand)
        upper_mid = lerp(shoulder, elbow, 0.5)
        forearm_mid = lerp(elbow, hand, 0.5)
        upper_vec = sub(elbow, shoulder)
        forearm_vec = sub(hand, elbow)
        set_pose("operator_right_upper_arm_anim", upper_mid, quat_from_z_axis_to_vector(upper_vec))
        set_pose("operator_right_hand_anim", hand, (0.0, 0.0, 0.0, 1.0))
        set_pose("operator_right_forearm_anim", forearm_mid, quat_from_z_axis_to_vector(forearm_vec))
        time.sleep(seconds / steps)


def run_cycle(hz):
    shoulder = (-0.72, -0.72, 1.16)
    start = (-0.45, -0.96, 1.08)
    pregrasp = (0.10, -0.94, 1.15)
    grasp = (0.33, -0.87, 1.15)
    lift = (0.31, -0.86, 1.23)
    carry = (0.18, -0.93, 1.27)

    path = [
        ("reach", start, pregrasp, 2.2),
        ("close", pregrasp, grasp, 1.2),
        ("lift", grasp, lift, 0.9),
        ("carry", lift, carry, 1.2),
        ("return", carry, lift, 0.7),
        ("lower", lift, grasp, 0.9),
        ("release", grasp, pregrasp, 1.0),
        ("retract", pregrasp, start, 1.4),
    ]

    elbow = bent_elbow(shoulder, start)
    set_pose("operator_right_upper_arm_anim", lerp(shoulder, elbow, 0.5), quat_from_z_axis_to_vector(sub(elbow, shoulder)))
    set_pose("operator_right_hand_anim", start, (0.0, 0.0, 0.0, 1.0))
    set_pose("operator_right_forearm_anim", lerp(elbow, start, 0.5), quat_from_z_axis_to_vector(sub(start, elbow)))
    for name, source, target, seconds in path:
        print(f"segment={name}", flush=True)
        move_segment(name, shoulder, source, target, seconds, hz)
    time.sleep(0.4)


def main():
    parser = argparse.ArgumentParser(
        description="Loop the mannequin right hand through a cup grasp motion in Gazebo."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of grasp cycles to run. 0 means loop until the process is stopped.",
    )
    parser.add_argument("--hz", type=float, default=3.0, help="Pose update rate.")
    args = parser.parse_args()

    cycle = 0
    while args.cycles <= 0 or cycle < args.cycles:
        cycle += 1
        print(f"cycle={cycle}", flush=True)
        run_cycle(args.hz)


if __name__ == "__main__":
    main()
