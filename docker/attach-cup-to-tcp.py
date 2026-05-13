#!/usr/bin/env python3
import argparse
import math
import subprocess
import time

import rclpy
import tf2_ros


GZ = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
ROBOT_WORLD_POS = (-0.2, -0.5, 1.021)
ROBOT_WORLD_YAW = 1.571
CUP_HOME = (0.34, -0.93, 1.065)
CUP_LIFT = (0.33, -0.88, 1.29)


def smoothstep(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def lerp(a, b, alpha):
    return tuple((1.0 - alpha) * x + alpha * y for x, y in zip(a, b))


def base_to_world(point):
    x, y, z = point
    c = math.cos(ROBOT_WORLD_YAW)
    s = math.sin(ROBOT_WORLD_YAW)
    return (
        ROBOT_WORLD_POS[0] + c * x - s * y,
        ROBOT_WORLD_POS[1] + s * x + c * y,
        ROBOT_WORLD_POS[2] + z,
    )


def set_pose(name, position, quat=(0.0, 0.0, 0.0, 1.0)):
    x, y, z = position
    qx, qy, qz, qw = quat
    req = (
        f'pose: {{name: "{name}" '
        f'position: {{x: {x:.5f} y: {y:.5f} z: {z:.5f}}} '
        f'orientation: {{x: {qx:.6f} y: {qy:.6f} z: {qz:.6f} w: {qw:.6f}}}}}'
    )
    result = subprocess.run(
        [
            GZ,
            "service",
            "-s",
            "/world/default/set_pose_vector/blocking",
            "--reqtype",
            "gz.msgs.Pose_V",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            req,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0 and "data: true" in result.stdout:
        return 0
    return result.returncode or 1


def tcp_world_from_buffer(buffer):
    transform = buffer.lookup_transform("link_base", "link_tcp", rclpy.time.Time())
    t = transform.transform.translation
    return base_to_world((t.x, t.y, t.z))


def main():
    parser = argparse.ArgumentParser(description="Keep red_cup_target attached to xArm7 link_tcp.")
    parser.add_argument("--delay", type=float, default=6.4, help="Seconds to wait before attaching.")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to keep the cup attached.")
    parser.add_argument("--hz", type=float, default=12.0)
    parser.add_argument("--reset", action="store_true", help="Reset the cup to its table pose and exit.")
    parser.add_argument("--offset-x", type=float, default=0.026)
    parser.add_argument("--offset-y", type=float, default=0.002)
    parser.add_argument("--offset-z", type=float, default=-0.075)
    parser.add_argument("--wait-near-cup", action="store_true", help="Attach only after TCP reaches the cup grasp pose.")
    parser.add_argument("--max-wait", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.012)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--planned-lift", action="store_true", help="After contact, move the cup along the planned grasp-lift path.")
    parser.add_argument("--lift-delay", type=float, default=0.8)
    parser.add_argument("--lift-duration", type=float, default=2.8)
    args = parser.parse_args()

    if args.reset:
        return set_pose("red_cup_target", CUP_HOME)

    rclpy.init()
    node = rclpy.create_node("attach_cup_to_tcp")
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer, node)
    offset = (args.offset_x, args.offset_y, args.offset_z)
    target_tcp = (
        CUP_HOME[0] - offset[0],
        CUP_HOME[1] - offset[1],
        CUP_HOME[2] - offset[2],
    )
    if args.wait_near_cup:
        deadline = time.monotonic() + args.max_wait
        wait_started = time.monotonic()
        min_distance = math.inf
        stable_samples = 0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                tcp_world = tcp_world_from_buffer(buffer)
                distance = math.dist(tcp_world, target_tcp)
                min_distance = min(min_distance, distance)
                if distance <= args.threshold:
                    stable_samples += 1
                else:
                    stable_samples = 0
                if stable_samples >= max(1, args.stable_samples):
                    print(
                        "cup_attach_triggered "
                        f"elapsed={time.monotonic() - wait_started:.2f} "
                        f"distance={distance:.4f} threshold={args.threshold} "
                        f"stable_samples={stable_samples}"
                    )
                    break
            except Exception:
                pass
        else:
            node.destroy_node()
            rclpy.shutdown()
            print(
                "cup_attach_updates=0 failures=1 "
                f"last_error=tcp_never_reached_cup threshold={args.threshold} "
                f"min_distance={min_distance:.4f}"
            )
            return 3
    else:
        deadline = time.monotonic() + args.delay
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

    end_time = time.monotonic() + args.duration
    period = 1.0 / max(1.0, args.hz)
    updates = 0
    failures = 0
    last_error = ""
    try:
        while rclpy.ok() and time.monotonic() < end_time:
            started = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.01)
            try:
                elapsed = started - (end_time - args.duration)
                if args.planned_lift:
                    if elapsed < args.lift_delay:
                        cup_position = CUP_HOME
                    else:
                        alpha = smoothstep((elapsed - args.lift_delay) / max(0.1, args.lift_duration))
                        cup_position = lerp(CUP_HOME, CUP_LIFT, alpha)
                else:
                    tcp_world = tcp_world_from_buffer(buffer)
                    cup_position = (
                        tcp_world[0] + offset[0],
                        tcp_world[1] + offset[1],
                        tcp_world[2] + offset[2],
                    )
                code = set_pose("red_cup_target", cup_position)
                if code == 0:
                    updates += 1
                else:
                    failures += 1
                    last_error = f"set_pose_return={code}"
            except Exception as exc:
                failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, period - elapsed))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"cup_attach_updates={updates} failures={failures} last_error={last_error}")
    return 0 if updates > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
