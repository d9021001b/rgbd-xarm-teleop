#!/usr/bin/env python3
import sys

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint


def main():
    if len(sys.argv) < 8:
        print("usage: send-xarm-goal.py j1 j2 j3 j4 j5 j6 j7 [seconds]", file=sys.stderr)
        return 2

    positions = [float(x) for x in sys.argv[1:8]]
    seconds = float(sys.argv[8]) if len(sys.argv) > 8 else 3.0

    rclpy.init()
    node = rclpy.create_node("send_xarm_goal_once")
    client = ActionClient(node, FollowJointTrajectory, "/xarm7_traj_controller/follow_joint_trajectory")
    if not client.wait_for_server(timeout_sec=5.0):
        print("action server unavailable", file=sys.stderr)
        return 3

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [f"joint{i}" for i in range(1, 8)]
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0.0] * 7
    point.time_from_start.sec = int(seconds)
    point.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
    goal.trajectory.points = [point]

    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if not handle.accepted:
        print("goal rejected", file=sys.stderr)
        return 4

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    print(result.error_code)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result.error_code == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
