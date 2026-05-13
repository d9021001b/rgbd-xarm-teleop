#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
BASE_LINK = "link_base"
TIP_LINK = "link_tcp"

ROBOT_WORLD_POS = np.array([-0.2, -0.5, 1.021], dtype=float)
ROBOT_WORLD_YAW = 1.571

SMPLX_KEYPOINTS_WORLD = {
    "ready": np.array([0.430, -1.432, 1.155], dtype=float),
    "hover": np.array([0.320, -0.915, 1.203], dtype=float),
    "grasp": np.array([0.314, -0.932, 1.147], dtype=float),
    "lift": np.array([0.306, -0.883, 1.351], dtype=float),
}
CUP_HOME_WORLD = np.array([0.340, -0.930, 1.065], dtype=float)
# link_tcp is the visual center of the four-finger opening. It only lines up
# with the cup if the IK also constrains the gripper orientation downward.
TCP_GRASP_WORLD = CUP_HOME_WORLD + np.array([0.000, 0.000, 0.105], dtype=float)
CUP_VISUAL_GRASP_OFFSET_WORLD = np.array([0.000, 0.000, 0.000], dtype=float)
CUP_TASK_KEYPOINTS_WORLD = {
    "ready": np.array([0.330, -1.080, 1.245], dtype=float),
    "hover": TCP_GRASP_WORLD + CUP_VISUAL_GRASP_OFFSET_WORLD + np.array([0.000, 0.000, 0.240], dtype=float),
    "grasp": TCP_GRASP_WORLD + CUP_VISUAL_GRASP_OFFSET_WORLD + np.array([0.000, 0.000, -0.030], dtype=float),
    "lift": TCP_GRASP_WORLD + CUP_VISUAL_GRASP_OFFSET_WORLD + np.array([0.000, 0.000, 0.300], dtype=float),
}
CUP_TASK_KEYPOINTS_BASE = {}
RETARGET_RUNTIME_CONFIG = {}
LOADED_TRAJECTORY_METADATA = {}


def default_retarget_config():
    return {
        "schema": "xarm7_smplx_retarget_calibration.v1",
        "frames": {
            "world": "gazebo_default_world",
            "robot_base": BASE_LINK,
            "target_tip": TIP_LINK,
        },
        "robot_world": {
            "position": ROBOT_WORLD_POS.tolist(),
            "yaw": ROBOT_WORLD_YAW,
        },
        "cup_world": CUP_HOME_WORLD.tolist(),
        "task": {
            "mode": "world_waypoints",
            "ready_world": CUP_TASK_KEYPOINTS_WORLD["ready"].tolist(),
            "tcp_grasp_offset_world": [0.0, 0.0, 0.105],
            "visual_grasp_offset_world": CUP_VISUAL_GRASP_OFFSET_WORLD.tolist(),
            "hover_offset_from_tcp_grasp_world": [0.0, 0.0, 0.24],
            "grasp_offset_from_tcp_grasp_world": [0.0, 0.0, -0.03],
            "lift_offset_from_tcp_grasp_world": [0.0, 0.0, 0.3],
            "waypoints_base_from_tcp_grasp": {
                "ready": [-0.17, 0.0, 0.08],
                "hover": [0.0, 0.0, 0.08],
                "grasp": [0.0, 0.0, -0.03],
                "lift": [0.0, 0.0, 0.18],
            },
        },
        "orientation": {
            "mode": "tcp_z_down",
            "rotation_base": [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        },
        "target_limits_base": {
            "min": [-0.72, -0.68, 0.1],
            "max": [0.45, 0.25, None],
        },
        "ik": {
            "controller_blend_in_seconds": 1.0,
            "max_iterations": 80,
            "position_tolerance": 0.008,
            "orientation_tolerance": 0.08,
            "orientation_weight": 0.35,
        },
        "execution": {
            "mode": "keyframes",
            "segments": [
                {"name": "settle-ready", "to": "ready", "duration_seconds": 1.0},
                {"name": "approach-hover", "to": "hover", "duration_seconds": 3.5},
                {"name": "descend-grasp", "to": "grasp", "duration_seconds": 2.0},
                {"name": "hold-grasp", "to": "grasp", "duration_seconds": 0.7},
                {"name": "vertical-lift", "to": "lift", "duration_seconds": 2.5},
            ],
            "milestones": [
                {"name": "hover-over-cup", "time_seconds": 4.8, "duration_seconds": 6.0},
                {"name": "descend-to-cup", "time_seconds": 7.4, "duration_seconds": 5.0},
                {"name": "grasp-hold", "time_seconds": 8.0, "duration_seconds": 2.0},
                {"name": "vertical-lift", "time_seconds": "end", "duration_seconds": 6.0},
            ],
        },
    }


def deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def vector3(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise RuntimeError(f"{name} must be a 3-number vector")
    return result


def load_retarget_config(path):
    config = default_retarget_config()
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        deep_update(config, payload)
    return config


def apply_retarget_config(config):
    global BASE_LINK
    global TIP_LINK
    global ROBOT_WORLD_POS
    global ROBOT_WORLD_YAW
    global CUP_HOME_WORLD
    global TCP_GRASP_WORLD
    global CUP_VISUAL_GRASP_OFFSET_WORLD
    global CUP_TASK_KEYPOINTS_WORLD
    global CUP_TASK_KEYPOINTS_BASE
    global RETARGET_RUNTIME_CONFIG

    frames = config.get("frames", {})
    BASE_LINK = frames.get("robot_base", frames.get("base_link", BASE_LINK))
    TIP_LINK = frames.get("target_tip", frames.get("tip_link", TIP_LINK))

    robot_world = config.get("robot_world", {})
    ROBOT_WORLD_POS = vector3(robot_world.get("position", ROBOT_WORLD_POS), "robot_world.position")
    ROBOT_WORLD_YAW = float(robot_world.get("yaw", ROBOT_WORLD_YAW))

    task = config.get("task", {})
    CUP_HOME_WORLD = vector3(config.get("cup_world", CUP_HOME_WORLD), "cup_world")
    TCP_GRASP_WORLD = CUP_HOME_WORLD + vector3(
        task.get("tcp_grasp_offset_world", [0.0, 0.0, 0.105]),
        "task.tcp_grasp_offset_world",
    )
    CUP_VISUAL_GRASP_OFFSET_WORLD = vector3(
        task.get("visual_grasp_offset_world", [0.0, 0.0, 0.0]),
        "task.visual_grasp_offset_world",
    )
    visual_tcp_grasp = TCP_GRASP_WORLD + CUP_VISUAL_GRASP_OFFSET_WORLD
    if task.get("mode") == "cup_base_relative":
        tcp_grasp_base = world_to_robot_base(visual_tcp_grasp)
        offsets = task.get("waypoints_base_from_tcp_grasp", {})
        waypoint_names = tuple(offsets.keys()) if offsets else ("ready", "hover", "grasp", "lift")
        CUP_TASK_KEYPOINTS_BASE = {
            name: tcp_grasp_base + vector3(offsets.get(name, [0.0, 0.0, 0.0]), f"task.waypoints_base_from_tcp_grasp.{name}")
            for name in waypoint_names
        }
        CUP_TASK_KEYPOINTS_WORLD = {
            name: robot_base_to_world(point_base) for name, point_base in CUP_TASK_KEYPOINTS_BASE.items()
        }
    else:
        CUP_TASK_KEYPOINTS_WORLD = {
            "ready": vector3(task.get("ready_world", [0.330, -1.080, 1.245]), "task.ready_world"),
            "hover": visual_tcp_grasp
            + vector3(task.get("hover_offset_from_tcp_grasp_world", [0.0, 0.0, 0.24]), "task.hover_offset_from_tcp_grasp_world"),
            "grasp": visual_tcp_grasp
            + vector3(task.get("grasp_offset_from_tcp_grasp_world", [0.0, 0.0, -0.03]), "task.grasp_offset_from_tcp_grasp_world"),
            "lift": visual_tcp_grasp
            + vector3(task.get("lift_offset_from_tcp_grasp_world", [0.0, 0.0, 0.3]), "task.lift_offset_from_tcp_grasp_world"),
        }
        CUP_TASK_KEYPOINTS_BASE = {name: retarget_target(point) for name, point in CUP_TASK_KEYPOINTS_WORLD.items()}
    RETARGET_RUNTIME_CONFIG = config


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def rot_axis(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(1e-12, np.linalg.norm(axis))
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def wrap_angle(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def wrap_angles(values):
    values = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(values), np.cos(values))


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


def functional_min_projection_norm():
    return float(RETARGET_RUNTIME_CONFIG.get("functional", {}).get("min_projection_norm", 1e-6))


def projected_angle_mask(vector, min_projection_norm=None):
    if min_projection_norm is None:
        min_projection_norm = functional_min_projection_norm()
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray(
        [
            1.0 if math.hypot(x, y) >= min_projection_norm else 0.0,
            1.0 if math.hypot(x, z) >= min_projection_norm else 0.0,
            1.0 if math.hypot(y, z) >= min_projection_norm else 0.0,
        ],
        dtype=float,
    )


def projected_angle_metrics(source_vector, robot_vector):
    mask = projected_angle_mask(source_vector) * projected_angle_mask(robot_vector)
    error = wrap_angles(projected_angles(source_vector) - projected_angles(robot_vector)) * mask
    return {
        "rad": error,
        "deg": np.degrees(error),
        "abs_deg": np.abs(np.degrees(error)),
        "max_rad": float(np.max(np.abs(error))),
        "max_deg": float(np.max(np.abs(np.degrees(error)))),
    }


def functional_mapping_metrics(
    smplx_shoulder_base,
    smplx_elbow_base,
    smplx_wrist_base,
    robot_shoulder_anchor,
    robot_elbow_base,
    robot_tcp_base,
):
    source_rotation = functional_source_vector_rotation_base()
    smplx_forearm = source_rotation @ (
        np.asarray(smplx_wrist_base, dtype=float) - np.asarray(smplx_elbow_base, dtype=float)
    )
    smplx_upper = source_rotation @ (
        np.asarray(smplx_elbow_base, dtype=float) - np.asarray(smplx_shoulder_base, dtype=float)
    )
    robot_forearm = np.asarray(robot_tcp_base, dtype=float) - np.asarray(robot_elbow_base, dtype=float)
    robot_upper = np.asarray(robot_elbow_base, dtype=float) - np.asarray(robot_shoulder_anchor, dtype=float)
    forearm = projected_angle_metrics(smplx_forearm, robot_forearm)
    upper = projected_angle_metrics(smplx_upper, robot_upper)
    smplx_included = included_angle(smplx_upper, smplx_forearm)
    robot_included = included_angle(robot_upper, robot_forearm)
    included_error = abs(smplx_included - robot_included)
    return {
        "forearm_projected_rad": forearm["rad"],
        "forearm_projected_deg": forearm["deg"],
        "forearm_projected_abs_deg": forearm["abs_deg"],
        "forearm_max_rad": forearm["max_rad"],
        "forearm_max_deg": forearm["max_deg"],
        "upper_arm_projected_rad": upper["rad"],
        "upper_arm_projected_deg": upper["deg"],
        "upper_arm_projected_abs_deg": upper["abs_deg"],
        "upper_arm_max_rad": upper["max_rad"],
        "upper_arm_max_deg": upper["max_deg"],
        "included_angle_smplx_deg": math.degrees(smplx_included),
        "included_angle_robot_deg": math.degrees(robot_included),
        "included_angle_abs_error_rad": included_error,
        "included_angle_abs_error_deg": math.degrees(included_error),
    }


def rpy_matrix(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def make_transform(xyz=None, rpy=None):
    transform = np.eye(4, dtype=float)
    if rpy is not None:
        transform[:3, :3] = rpy_matrix(rpy)
    if xyz is not None:
        transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def parse_vector(text, default):
    if text is None:
        return list(default)
    return [float(part) for part in text.split()]


def clean_robot_description(text):
    prefix = "String value is: "
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class KinematicChain:
    def __init__(self, urdf_text, base_link=BASE_LINK, tip_link=TIP_LINK):
        root = ET.fromstring(urdf_text)
        joints_by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            joints_by_child[child] = joint

        chain = []
        link = tip_link
        while link != base_link:
            if link not in joints_by_child:
                raise RuntimeError(f"Could not find joint from {base_link} to {tip_link}; stopped at {link}")
            joint = joints_by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        chain.reverse()

        self.segments = []
        self.limits = []
        for joint in chain:
            origin = joint.find("origin")
            xyz = parse_vector(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0])
            rpy = parse_vector(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0])
            axis_node = joint.find("axis")
            axis = parse_vector(axis_node.attrib.get("xyz") if axis_node is not None else None, [0, 0, 1])
            limit = joint.find("limit")
            lower, upper = -math.inf, math.inf
            if limit is not None:
                lower = float(limit.attrib.get("lower", lower))
                upper = float(limit.attrib.get("upper", upper))
            self.segments.append(
                {
                    "name": joint.attrib["name"],
                    "type": joint.attrib["type"],
                    "origin": make_transform(xyz, rpy),
                    "axis": np.asarray(axis, dtype=float),
                }
            )
            if joint.attrib["type"] in ("revolute", "continuous"):
                self.limits.append((lower, upper))

        if [seg["name"] for seg in self.segments if seg["type"] in ("revolute", "continuous")] != JOINT_NAMES:
            found = [seg["name"] for seg in self.segments if seg["type"] in ("revolute", "continuous")]
            raise RuntimeError(f"Unexpected joint chain: {found}")
        self.limits = np.asarray(self.limits, dtype=float)

    @staticmethod
    def damped_pseudoinverse(jacobian, damping):
        rows = jacobian.shape[0]
        lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(rows)
        return jacobian.T @ np.linalg.solve(lhs, np.eye(rows))

    def fk(self, q):
        transform = np.eye(4, dtype=float)
        joint_positions = []
        joint_axes = []
        q_index = 0
        for segment in self.segments:
            transform = transform @ segment["origin"]
            if segment["type"] in ("revolute", "continuous"):
                axis_world = transform[:3, :3] @ segment["axis"]
                joint_axes.append(axis_world)
                joint_positions.append(transform[:3, 3].copy())
                rotation = np.eye(4, dtype=float)
                rotation[:3, :3] = rot_axis(segment["axis"], q[q_index])
                transform = transform @ rotation
                q_index += 1
        return transform, np.asarray(joint_positions), np.asarray(joint_axes)

    def solve_position(self, target, seed, iterations=240, damping=0.045, tolerance=0.006):
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(iterations):
            transform, joint_positions, joint_axes = self.fk(q)
            tip = transform[:3, 3]
            error = np.asarray(target, dtype=float) - tip
            if np.linalg.norm(error) < tolerance:
                return q, True, float(np.linalg.norm(error))
            jacobian = np.zeros((3, len(q)), dtype=float)
            for idx, (origin, axis) in enumerate(zip(joint_positions, joint_axes)):
                jacobian[:, idx] = np.cross(axis, tip - origin)
            lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(3)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            max_step = np.max(np.abs(dq))
            if max_step > 0.08:
                dq *= 0.08 / max_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)
        transform, _, _ = self.fk(q)
        return q, False, float(np.linalg.norm(np.asarray(target, dtype=float) - transform[:3, 3]))

    def solve_pose(
        self,
        target,
        desired_rotation,
        seed,
        iterations=360,
        damping=0.055,
        position_tolerance=0.008,
        orientation_tolerance=0.08,
        orientation_weight=0.35,
    ):
        q = np.asarray(seed, dtype=float).copy()
        desired_rotation = np.asarray(desired_rotation, dtype=float)
        for _ in range(iterations):
            transform, joint_positions, joint_axes = self.fk(q)
            tip = transform[:3, 3]
            rotation = transform[:3, :3]
            position_error = np.asarray(target, dtype=float) - tip
            orientation_error = 0.5 * (
                np.cross(rotation[:, 0], desired_rotation[:, 0])
                + np.cross(rotation[:, 1], desired_rotation[:, 1])
                + np.cross(rotation[:, 2], desired_rotation[:, 2])
            )
            if (
                np.linalg.norm(position_error) < position_tolerance
                and np.linalg.norm(orientation_error) < orientation_tolerance
            ):
                return q, True, float(np.linalg.norm(position_error)), float(np.linalg.norm(orientation_error))
            jacobian = np.zeros((6, len(q)), dtype=float)
            for idx, (origin, axis) in enumerate(zip(joint_positions, joint_axes)):
                jacobian[:3, idx] = np.cross(axis, tip - origin)
                jacobian[3:, idx] = axis
            error = np.concatenate([position_error, orientation_weight * orientation_error])
            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:, :] *= orientation_weight
            lhs = weighted_jacobian @ weighted_jacobian.T + (damping * damping) * np.eye(6)
            dq = weighted_jacobian.T @ np.linalg.solve(lhs, error)
            max_step = np.max(np.abs(dq))
            if max_step > 0.06:
                dq *= 0.06 / max_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)
        transform, _, _ = self.fk(q)
        position_error = np.asarray(target, dtype=float) - transform[:3, 3]
        rotation = transform[:3, :3]
        orientation_error = 0.5 * (
            np.cross(rotation[:, 0], desired_rotation[:, 0])
            + np.cross(rotation[:, 1], desired_rotation[:, 1])
            + np.cross(rotation[:, 2], desired_rotation[:, 2])
        )
        return (
            q,
            False,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(orientation_error)),
        )

    def solve_full_arm(
        self,
        target,
        desired_rotation,
        elbow_target,
        seed,
        elbow_joint_index=3,
        iterations=420,
        damping=0.06,
        position_tolerance=0.010,
        orientation_tolerance=0.12,
        elbow_tolerance=0.055,
        orientation_weight=0.12,
        elbow_weight=0.30,
    ):
        q = np.asarray(seed, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        desired_rotation = np.asarray(desired_rotation, dtype=float)
        elbow_target = np.asarray(elbow_target, dtype=float)
        elbow_joint_index = int(elbow_joint_index)
        for _ in range(iterations):
            transform, joint_positions, joint_axes = self.fk(q)
            tip = transform[:3, 3]
            rotation = transform[:3, :3]
            elbow = joint_positions[elbow_joint_index]
            position_error = target - tip
            elbow_error = elbow_target - elbow
            orientation_error = 0.5 * (
                np.cross(rotation[:, 0], desired_rotation[:, 0])
                + np.cross(rotation[:, 1], desired_rotation[:, 1])
                + np.cross(rotation[:, 2], desired_rotation[:, 2])
            )
            if (
                np.linalg.norm(position_error) < position_tolerance
                and np.linalg.norm(orientation_error) < orientation_tolerance
                and np.linalg.norm(elbow_error) < elbow_tolerance
            ):
                return (
                    q,
                    True,
                    float(np.linalg.norm(position_error)),
                    float(np.linalg.norm(orientation_error)),
                    float(np.linalg.norm(elbow_error)),
                )
            jacobian = np.zeros((9, len(q)), dtype=float)
            for idx, (origin, axis) in enumerate(zip(joint_positions, joint_axes)):
                jacobian[:3, idx] = np.cross(axis, tip - origin)
                jacobian[3:6, idx] = axis
                if idx < elbow_joint_index:
                    jacobian[6:9, idx] = np.cross(axis, elbow - origin)
            error = np.concatenate(
                [
                    position_error,
                    orientation_weight * orientation_error,
                    elbow_weight * elbow_error,
                ]
            )
            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:6, :] *= orientation_weight
            weighted_jacobian[6:9, :] *= elbow_weight
            lhs = weighted_jacobian @ weighted_jacobian.T + (damping * damping) * np.eye(9)
            dq = weighted_jacobian.T @ np.linalg.solve(lhs, error)
            max_step = np.max(np.abs(dq))
            if max_step > 0.055:
                dq *= 0.055 / max_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)
        transform, joint_positions, _ = self.fk(q)
        elbow = joint_positions[elbow_joint_index]
        position_error = target - transform[:3, 3]
        rotation = transform[:3, :3]
        orientation_error = 0.5 * (
            np.cross(rotation[:, 0], desired_rotation[:, 0])
            + np.cross(rotation[:, 1], desired_rotation[:, 1])
            + np.cross(rotation[:, 2], desired_rotation[:, 2])
        )
        elbow_error = elbow_target - elbow
        return (
            q,
            False,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(orientation_error)),
            float(np.linalg.norm(elbow_error)),
        )

    def solve_full_arm_hierarchical(
        self,
        target,
        desired_rotation,
        elbow_target,
        seed,
        elbow_joint_index=3,
        iterations=420,
        primary_damping=0.040,
        secondary_damping=0.070,
        position_tolerance=0.010,
        orientation_tolerance=0.22,
        elbow_tolerance=0.080,
        orientation_weight=0.030,
        elbow_weight=0.120,
        secondary_step_scale=1.0,
        max_step=0.055,
    ):
        q = np.asarray(seed, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        desired_rotation = np.asarray(desired_rotation, dtype=float)
        elbow_target = np.asarray(elbow_target, dtype=float)
        elbow_joint_index = int(elbow_joint_index)
        joint_count = len(q)
        identity = np.eye(joint_count)
        for _ in range(iterations):
            transform, joint_positions, joint_axes = self.fk(q)
            tip = transform[:3, 3]
            rotation = transform[:3, :3]
            elbow = joint_positions[elbow_joint_index]
            position_error = target - tip
            elbow_error = elbow_target - elbow
            orientation_error = 0.5 * (
                np.cross(rotation[:, 0], desired_rotation[:, 0])
                + np.cross(rotation[:, 1], desired_rotation[:, 1])
                + np.cross(rotation[:, 2], desired_rotation[:, 2])
            )
            if (
                np.linalg.norm(position_error) < position_tolerance
                and np.linalg.norm(orientation_error) < orientation_tolerance
                and np.linalg.norm(elbow_error) < elbow_tolerance
            ):
                return (
                    q,
                    True,
                    float(np.linalg.norm(position_error)),
                    float(np.linalg.norm(orientation_error)),
                    float(np.linalg.norm(elbow_error)),
                )

            wrist_jacobian = np.zeros((3, joint_count), dtype=float)
            orientation_jacobian = np.zeros((3, joint_count), dtype=float)
            elbow_jacobian = np.zeros((3, joint_count), dtype=float)
            for idx, (origin, axis) in enumerate(zip(joint_positions, joint_axes)):
                wrist_jacobian[:, idx] = np.cross(axis, tip - origin)
                orientation_jacobian[:, idx] = axis
                if idx < elbow_joint_index:
                    elbow_jacobian[:, idx] = np.cross(axis, elbow - origin)

            if orientation_weight > 0.0:
                primary_jacobian = np.vstack(
                    [
                        wrist_jacobian,
                        orientation_weight * orientation_jacobian,
                    ]
                )
                primary_error = np.concatenate(
                    [
                        position_error,
                        orientation_weight * orientation_error,
                    ]
                )
            else:
                primary_jacobian = wrist_jacobian
                primary_error = position_error

            primary_pinv = self.damped_pseudoinverse(primary_jacobian, primary_damping)
            dq_primary = primary_pinv @ primary_error
            primary_pinv_for_projection = np.linalg.pinv(primary_jacobian, rcond=1e-4)
            nullspace = identity - primary_pinv_for_projection @ primary_jacobian

            secondary_jacobians = []
            secondary_errors = []
            if elbow_weight > 0.0:
                secondary_jacobians.append(elbow_weight * elbow_jacobian)
                secondary_errors.append(elbow_weight * elbow_error)

            dq_secondary = np.zeros(joint_count, dtype=float)
            position_norm = float(np.linalg.norm(position_error))
            secondary_guard = max(0.030, position_tolerance * 2.0)
            orientation_norm = float(np.linalg.norm(orientation_error))
            orientation_guard = max(0.25, orientation_tolerance * 2.0)
            primary_is_close = (
                position_norm < secondary_guard
                and (orientation_weight <= 0.0 or orientation_norm < orientation_guard)
            )
            if secondary_jacobians and primary_is_close:
                secondary_jacobian = np.vstack(secondary_jacobians)
                secondary_error = np.concatenate(secondary_errors)
                secondary_residual = secondary_error - secondary_jacobian @ dq_primary
                projected_secondary = secondary_jacobian @ nullspace
                projected_pinv = self.damped_pseudoinverse(projected_secondary, secondary_damping)
                dq_secondary = nullspace @ (projected_pinv @ secondary_residual)
                secondary_gate = 1.0 - (position_norm / secondary_guard)
                dq_secondary *= secondary_step_scale * secondary_gate

            dq = dq_primary + dq_secondary
            largest_step = np.max(np.abs(dq))
            if largest_step > max_step:
                dq *= max_step / largest_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)

        transform, joint_positions, _ = self.fk(q)
        elbow = joint_positions[elbow_joint_index]
        position_error = target - transform[:3, 3]
        rotation = transform[:3, :3]
        orientation_error = 0.5 * (
            np.cross(rotation[:, 0], desired_rotation[:, 0])
            + np.cross(rotation[:, 1], desired_rotation[:, 1])
            + np.cross(rotation[:, 2], desired_rotation[:, 2])
        )
        elbow_error = elbow_target - elbow
        return (
            q,
            False,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(orientation_error)),
            float(np.linalg.norm(elbow_error)),
        )

    def functional_angle_state(self, q, shoulder_anchor, elbow_joint_index=3, shoulder_joint_index=None):
        transform, joint_positions, _ = self.fk(q)
        tcp = transform[:3, 3]
        elbow = joint_positions[int(elbow_joint_index)]
        if shoulder_joint_index is None:
            shoulder = np.asarray(shoulder_anchor, dtype=float)
        else:
            shoulder = joint_positions[int(shoulder_joint_index)]
        forearm = tcp - elbow
        upper_arm = elbow - shoulder
        return np.concatenate([projected_angles(forearm), projected_angles(upper_arm)])

    def functional_included_angle_state(self, q, shoulder_anchor, elbow_joint_index=3, shoulder_joint_index=None):
        transform, joint_positions, _ = self.fk(q)
        tcp = transform[:3, 3]
        elbow = joint_positions[int(elbow_joint_index)]
        if shoulder_joint_index is None:
            shoulder = np.asarray(shoulder_anchor, dtype=float)
        else:
            shoulder = joint_positions[int(shoulder_joint_index)]
        return included_angle(elbow - shoulder, tcp - elbow)

    def solve_functional_hierarchical(
        self,
        target,
        wrist_base,
        elbow_base,
        shoulder_base,
        seed,
        robot_shoulder_anchor,
        elbow_joint_index=3,
        shoulder_joint_index=None,
        iterations=220,
        primary_damping=0.045,
        secondary_damping=0.075,
        position_tolerance=0.014,
        forearm_tolerance=0.35,
        upper_arm_tolerance=0.55,
        forearm_weight=1.0,
        upper_arm_weight=0.65,
        included_angle_weight=0.0,
        secondary_step_scale=0.70,
        max_step=0.055,
        finite_difference_step=1e-4,
        secondary_position_guard=0.035,
    ):
        q = np.asarray(seed, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        wrist_base = np.asarray(wrist_base, dtype=float)
        elbow_base = np.asarray(elbow_base, dtype=float)
        shoulder_base = np.asarray(shoulder_base, dtype=float)
        robot_shoulder_anchor = np.asarray(robot_shoulder_anchor, dtype=float)
        elbow_joint_index = int(elbow_joint_index)
        joint_count = len(q)
        identity = np.eye(joint_count)

        source_rotation = functional_source_vector_rotation_base()
        human_forearm = source_rotation @ (wrist_base - elbow_base)
        human_upper_arm = source_rotation @ (elbow_base - shoulder_base)
        target_included_angle = included_angle(human_upper_arm, human_forearm)
        target_angles = np.concatenate([projected_angles(human_forearm), projected_angles(human_upper_arm)])
        target_masks = np.concatenate(
            [projected_angle_mask(human_forearm), projected_angle_mask(human_upper_arm)]
        )
        angle_weights = np.asarray([forearm_weight] * 3 + [upper_arm_weight] * 3, dtype=float) * target_masks

        last_metrics = None
        for _ in range(iterations):
            transform, joint_positions, joint_axes = self.fk(q)
            tip = transform[:3, 3]
            position_error = target - tip
            current_angles = self.functional_angle_state(
                q,
                robot_shoulder_anchor,
                elbow_joint_index,
                shoulder_joint_index,
            )
            angle_error = wrap_angles(target_angles - current_angles) * target_masks
            forearm_error = angle_error[:3]
            upper_arm_error = angle_error[3:]
            included_angle_error = target_included_angle - self.functional_included_angle_state(
                q,
                robot_shoulder_anchor,
                elbow_joint_index,
                shoulder_joint_index,
            )
            last_metrics = {
                "forearm_projected_rad": forearm_error.copy(),
                "upper_arm_projected_rad": upper_arm_error.copy(),
                "forearm_max_rad": float(np.max(np.abs(forearm_error))),
                "upper_arm_max_rad": float(np.max(np.abs(upper_arm_error))),
                "included_angle_abs_error_rad": abs(float(included_angle_error)),
            }
            if (
                np.linalg.norm(position_error) < position_tolerance
                and last_metrics["forearm_max_rad"] < forearm_tolerance
                and last_metrics["upper_arm_max_rad"] < upper_arm_tolerance
            ):
                return (
                    q,
                    True,
                    float(np.linalg.norm(position_error)),
                    last_metrics["forearm_max_rad"],
                    last_metrics["upper_arm_max_rad"],
                    last_metrics,
                )

            wrist_jacobian = np.zeros((3, joint_count), dtype=float)
            for idx, (origin, axis) in enumerate(zip(joint_positions, joint_axes)):
                wrist_jacobian[:, idx] = np.cross(axis, tip - origin)
            wrist_pinv = self.damped_pseudoinverse(wrist_jacobian, primary_damping)
            dq_primary = wrist_pinv @ position_error
            wrist_pinv_for_projection = np.linalg.pinv(wrist_jacobian, rcond=1e-4)
            nullspace = identity - wrist_pinv_for_projection @ wrist_jacobian

            dq_secondary = np.zeros(joint_count, dtype=float)
            position_norm = float(np.linalg.norm(position_error))
            if position_norm < secondary_position_guard and np.any(angle_weights > 0.0):
                angle_jacobian = np.zeros((6, joint_count), dtype=float)
                base_state = current_angles
                for idx in range(joint_count):
                    q_eps = q.copy()
                    q_eps[idx] += finite_difference_step
                    state_eps = self.functional_angle_state(
                        q_eps,
                        robot_shoulder_anchor,
                        elbow_joint_index,
                        shoulder_joint_index,
                    )
                    angle_jacobian[:, idx] = wrap_angles(state_eps - base_state) / finite_difference_step
                weighted_jacobian = angle_jacobian * angle_weights[:, None]
                weighted_error = angle_error * angle_weights
                if included_angle_weight > 0.0:
                    base_included_angle = self.functional_included_angle_state(
                        q,
                        robot_shoulder_anchor,
                        elbow_joint_index,
                        shoulder_joint_index,
                    )
                    included_jacobian = np.zeros((1, joint_count), dtype=float)
                    for idx in range(joint_count):
                        q_eps = q.copy()
                        q_eps[idx] += finite_difference_step
                        included_jacobian[0, idx] = (
                            self.functional_included_angle_state(
                                q_eps,
                                robot_shoulder_anchor,
                                elbow_joint_index,
                                shoulder_joint_index,
                            )
                            - base_included_angle
                        ) / finite_difference_step
                    weighted_jacobian = np.vstack([weighted_jacobian, included_angle_weight * included_jacobian])
                    weighted_error = np.concatenate(
                        [weighted_error, np.asarray([included_angle_weight * included_angle_error], dtype=float)]
                    )
                secondary_residual = weighted_error - weighted_jacobian @ dq_primary
                projected_secondary = weighted_jacobian @ nullspace
                projected_pinv = self.damped_pseudoinverse(projected_secondary, secondary_damping)
                secondary_gate = 1.0 - (position_norm / secondary_position_guard)
                dq_secondary = nullspace @ (projected_pinv @ secondary_residual)
                dq_secondary *= secondary_step_scale * secondary_gate

            dq = dq_primary + dq_secondary
            largest_step = np.max(np.abs(dq))
            if largest_step > max_step:
                dq *= max_step / largest_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)

        transform, _, _ = self.fk(q)
        position_error = target - transform[:3, 3]
        if last_metrics is None:
            current_angles = self.functional_angle_state(
                q,
                robot_shoulder_anchor,
                elbow_joint_index,
                shoulder_joint_index,
            )
            angle_error = wrap_angles(target_angles - current_angles) * target_masks
            last_metrics = {
                "forearm_projected_rad": angle_error[:3],
                "upper_arm_projected_rad": angle_error[3:],
                "forearm_max_rad": float(np.max(np.abs(angle_error[:3]))),
                "upper_arm_max_rad": float(np.max(np.abs(angle_error[3:]))),
                "included_angle_abs_error_rad": abs(
                    float(
                        target_included_angle
                        - self.functional_included_angle_state(
                            q,
                            robot_shoulder_anchor,
                            elbow_joint_index,
                            shoulder_joint_index,
                        )
                    )
                ),
            }
        return (
            q,
            False,
            float(np.linalg.norm(position_error)),
            last_metrics["forearm_max_rad"],
            last_metrics["upper_arm_max_rad"],
            last_metrics,
        )

    def solve_table_edge_soft_functional_hierarchical(
        self,
        target,
        desired_rotation,
        wrist_base,
        elbow_base,
        shoulder_base,
        seed,
        robot_shoulder_anchor,
        elbow_joint_index=3,
        shoulder_joint_index=None,
        iterations=220,
        primary_damping=0.045,
        secondary_damping=0.075,
        position_tolerance=0.014,
        orientation_tolerance=0.30,
        forearm_tolerance=1.20,
        upper_arm_tolerance=1.45,
        anti_self_tolerance=0.02,
        orientation_weight=0.18,
        forearm_weight=0.20,
        upper_arm_weight=0.10,
        included_angle_weight=0.0,
        anti_self_weight=0.70,
        joint_regularization_weight=0.06,
        secondary_step_scale=0.65,
        max_step=0.055,
        finite_difference_step=1e-4,
        secondary_position_guard=0.045,
    ):
        q = np.asarray(seed, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        desired_rotation = np.asarray(desired_rotation, dtype=float)
        reference_q = q.copy()
        wrist_base = np.asarray(wrist_base, dtype=float)
        elbow_base = np.asarray(elbow_base, dtype=float)
        shoulder_base = np.asarray(shoulder_base, dtype=float)
        robot_shoulder_anchor = np.asarray(robot_shoulder_anchor, dtype=float)
        elbow_joint_index = int(elbow_joint_index)
        joint_count = len(q)
        identity = np.eye(joint_count)

        source_rotation = functional_source_vector_rotation_base()
        human_forearm = source_rotation @ (wrist_base - elbow_base)
        human_upper_arm = source_rotation @ (elbow_base - shoulder_base)
        target_included_angle = included_angle(human_upper_arm, human_forearm)
        target_angles = np.concatenate([projected_angles(human_forearm), projected_angles(human_upper_arm)])
        target_masks = np.concatenate(
            [projected_angle_mask(human_forearm), projected_angle_mask(human_upper_arm)]
        )
        angle_weights = np.asarray([forearm_weight] * 3 + [upper_arm_weight] * 3, dtype=float) * target_masks

        last_metrics = None
        for _ in range(iterations):
            transform, joint_positions, joint_axes = self.fk(q)
            tip = transform[:3, 3]
            rotation = transform[:3, :3]
            position_error = target - tip
            orientation_error = 0.5 * (
                np.cross(rotation[:, 0], desired_rotation[:, 0])
                + np.cross(rotation[:, 1], desired_rotation[:, 1])
                + np.cross(rotation[:, 2], desired_rotation[:, 2])
            )
            current_angles = self.functional_angle_state(
                q,
                robot_shoulder_anchor,
                elbow_joint_index,
                shoulder_joint_index,
            )
            angle_error = wrap_angles(target_angles - current_angles) * target_masks
            forearm_error = angle_error[:3]
            upper_arm_error = angle_error[3:]
            included_angle_error = target_included_angle - self.functional_included_angle_state(
                q,
                robot_shoulder_anchor,
                elbow_joint_index,
                shoulder_joint_index,
            )
            anti_metrics = anti_self_insertion_metrics_for_fk(tip, joint_positions, rotation)
            anti_penalty = float(anti_metrics.get("penalty", 0.0))
            last_metrics = {
                "forearm_projected_rad": forearm_error.copy(),
                "upper_arm_projected_rad": upper_arm_error.copy(),
                "forearm_max_rad": float(np.max(np.abs(forearm_error))),
                "upper_arm_max_rad": float(np.max(np.abs(upper_arm_error))),
                "included_angle_abs_error_rad": abs(float(included_angle_error)),
                "orientation_norm_rad": float(np.linalg.norm(orientation_error)),
                "anti_self_penalty": anti_penalty,
                "anti_self_tool_penalty": float(anti_metrics.get("tool_penalty", 0.0)),
                "anti_self_forearm_alignment": float(anti_metrics.get("forearm_alignment", 0.0)),
                "anti_self_tool_alignment": float(anti_metrics.get("tool_alignment", 0.0)),
            }

            if (
                np.linalg.norm(position_error) < position_tolerance
                and last_metrics["orientation_norm_rad"] < orientation_tolerance
                and last_metrics["forearm_max_rad"] < forearm_tolerance
                and last_metrics["upper_arm_max_rad"] < upper_arm_tolerance
                and anti_penalty <= anti_self_tolerance
            ):
                return (
                    q,
                    True,
                    float(np.linalg.norm(position_error)),
                    last_metrics["orientation_norm_rad"],
                    last_metrics["upper_arm_max_rad"],
                    last_metrics,
                )

            wrist_jacobian = np.zeros((3, joint_count), dtype=float)
            for idx, (origin, axis) in enumerate(zip(joint_positions, joint_axes)):
                wrist_jacobian[:, idx] = np.cross(axis, tip - origin)
            wrist_pinv = self.damped_pseudoinverse(wrist_jacobian, primary_damping)
            dq_primary = wrist_pinv @ position_error
            wrist_pinv_for_projection = np.linalg.pinv(wrist_jacobian, rcond=1e-4)
            nullspace = identity - wrist_pinv_for_projection @ wrist_jacobian

            dq_secondary = np.zeros(joint_count, dtype=float)
            position_norm = float(np.linalg.norm(position_error))
            if position_norm < secondary_position_guard:
                jacobian_blocks = []
                error_blocks = []

                if orientation_weight > 0.0:
                    orientation_jacobian = np.zeros((3, joint_count), dtype=float)
                    for idx, axis in enumerate(joint_axes):
                        orientation_jacobian[:, idx] = axis
                    jacobian_blocks.append(orientation_weight * orientation_jacobian)
                    error_blocks.append(orientation_weight * orientation_error)

                if np.any(angle_weights > 0.0):
                    angle_jacobian = np.zeros((6, joint_count), dtype=float)
                    base_state = current_angles
                    for idx in range(joint_count):
                        q_eps = q.copy()
                        q_eps[idx] += finite_difference_step
                        state_eps = self.functional_angle_state(
                            q_eps,
                            robot_shoulder_anchor,
                            elbow_joint_index,
                            shoulder_joint_index,
                        )
                        angle_jacobian[:, idx] = wrap_angles(state_eps - base_state) / finite_difference_step
                    jacobian_blocks.append(angle_jacobian * angle_weights[:, None])
                    error_blocks.append(angle_error * angle_weights)

                if included_angle_weight > 0.0:
                    base_included_angle = self.functional_included_angle_state(
                        q,
                        robot_shoulder_anchor,
                        elbow_joint_index,
                        shoulder_joint_index,
                    )
                    included_jacobian = np.zeros((1, joint_count), dtype=float)
                    for idx in range(joint_count):
                        q_eps = q.copy()
                        q_eps[idx] += finite_difference_step
                        included_jacobian[0, idx] = (
                            self.functional_included_angle_state(
                                q_eps,
                                robot_shoulder_anchor,
                                elbow_joint_index,
                                shoulder_joint_index,
                            )
                            - base_included_angle
                        ) / finite_difference_step
                    jacobian_blocks.append(included_angle_weight * included_jacobian)
                    error_blocks.append(np.asarray([included_angle_weight * included_angle_error], dtype=float))

                if anti_self_weight > 0.0 and anti_penalty > anti_self_tolerance:
                    penalty_jacobian = np.zeros((1, joint_count), dtype=float)
                    for idx in range(joint_count):
                        q_eps = q.copy()
                        q_eps[idx] += finite_difference_step
                        eps_transform, eps_joint_positions, _ = self.fk(q_eps)
                        eps_metrics = anti_self_insertion_metrics_for_fk(
                            eps_transform[:3, 3],
                            eps_joint_positions,
                            eps_transform[:3, :3],
                        )
                        penalty_jacobian[0, idx] = (
                            float(eps_metrics.get("penalty", 0.0)) - anti_penalty
                        ) / finite_difference_step
                    jacobian_blocks.append(anti_self_weight * penalty_jacobian)
                    error_blocks.append(np.asarray([-anti_self_weight * anti_penalty], dtype=float))

                if joint_regularization_weight > 0.0:
                    jacobian_blocks.append(joint_regularization_weight * identity)
                    error_blocks.append(joint_regularization_weight * (reference_q - q))

                if jacobian_blocks:
                    weighted_jacobian = np.vstack(jacobian_blocks)
                    weighted_error = np.concatenate(error_blocks)
                    secondary_residual = weighted_error - weighted_jacobian @ dq_primary
                    projected_secondary = weighted_jacobian @ nullspace
                    projected_pinv = self.damped_pseudoinverse(projected_secondary, secondary_damping)
                    secondary_gate = 1.0 - min(1.0, position_norm / max(secondary_position_guard, 1e-6))
                    dq_secondary = nullspace @ (projected_pinv @ secondary_residual)
                    dq_secondary *= secondary_step_scale * secondary_gate

            dq = dq_primary + dq_secondary
            largest_step = np.max(np.abs(dq))
            if largest_step > max_step:
                dq *= max_step / largest_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)

        transform, joint_positions, _ = self.fk(q)
        tip = transform[:3, 3]
        position_error = target - tip
        rotation = transform[:3, :3]
        orientation_error = 0.5 * (
            np.cross(rotation[:, 0], desired_rotation[:, 0])
            + np.cross(rotation[:, 1], desired_rotation[:, 1])
            + np.cross(rotation[:, 2], desired_rotation[:, 2])
        )
        current_angles = self.functional_angle_state(
            q,
            robot_shoulder_anchor,
            elbow_joint_index,
            shoulder_joint_index,
        )
        angle_error = wrap_angles(target_angles - current_angles) * target_masks
        anti_metrics = anti_self_insertion_metrics_for_fk(tip, joint_positions, transform[:3, :3])
        last_metrics = {
            "forearm_projected_rad": angle_error[:3],
            "upper_arm_projected_rad": angle_error[3:],
            "forearm_max_rad": float(np.max(np.abs(angle_error[:3]))),
            "upper_arm_max_rad": float(np.max(np.abs(angle_error[3:]))),
            "included_angle_abs_error_rad": abs(
                float(
                    target_included_angle
                    - self.functional_included_angle_state(
                        q,
                        robot_shoulder_anchor,
                        elbow_joint_index,
                        shoulder_joint_index,
                    )
                )
            ),
            "orientation_norm_rad": float(np.linalg.norm(orientation_error)),
            "anti_self_penalty": float(anti_metrics.get("penalty", 0.0)),
            "anti_self_tool_penalty": float(anti_metrics.get("tool_penalty", 0.0)),
            "anti_self_forearm_alignment": float(anti_metrics.get("forearm_alignment", 0.0)),
            "anti_self_tool_alignment": float(anti_metrics.get("tool_alignment", 0.0)),
        }
        return (
            q,
            False,
            float(np.linalg.norm(position_error)),
            last_metrics["orientation_norm_rad"],
            last_metrics["upper_arm_max_rad"],
            last_metrics,
        )

    def solve_ocra_baseline(
        self,
        target,
        desired_rotation,
        wrist_base,
        elbow_base,
        shoulder_base,
        seed,
        robot_shoulder_anchor,
        elbow_joint_index=3,
        shoulder_joint_index=None,
        iterations=180,
        damping=0.065,
        alpha=0.67,
        beta=0.33,
        target_position_weight=0.0,
        joint_regularization_weight=0.03,
        source_anchor_mode="absolute",
        robot_upper_arm_length=0.30,
        robot_forearm_length=0.34,
        position_tolerance=0.025,
        skeleton_tolerance=0.060,
        orientation_tolerance=0.45,
        max_step=0.060,
        finite_difference_step=1e-4,
    ):
        q = np.asarray(seed, dtype=float).copy()
        reference_q = q.copy()
        target = np.asarray(target, dtype=float)
        desired_rotation = np.asarray(desired_rotation, dtype=float)
        wrist_base = np.asarray(wrist_base, dtype=float)
        elbow_base = np.asarray(elbow_base, dtype=float)
        shoulder_base = np.asarray(shoulder_base, dtype=float)
        robot_shoulder_anchor = np.asarray(robot_shoulder_anchor, dtype=float)
        elbow_joint_index = int(elbow_joint_index)
        joint_count = len(q)

        alpha = max(0.0, float(alpha))
        beta = max(0.0, float(beta))
        total = alpha + beta
        if total <= 1e-9:
            alpha, beta = 1.0, 0.0
        else:
            alpha, beta = alpha / total, beta / total

        source_rotation = functional_source_vector_rotation_base()
        source_forearm = source_rotation @ (wrist_base - elbow_base)
        source_upper = source_rotation @ (elbow_base - shoulder_base)
        if source_anchor_mode == "robot_shoulder_scaled":
            source_shoulder = robot_shoulder_anchor.copy()
            source_elbow = source_shoulder + normalized(source_upper, [1.0, 0.0, 0.0]) * float(robot_upper_arm_length)
            source_wrist = source_elbow + normalized(source_forearm, [1.0, 0.0, 0.0]) * float(robot_forearm_length)
        elif source_anchor_mode == "absolute":
            source_wrist = target.copy()
            source_elbow = source_wrist - source_forearm
            source_shoulder = source_elbow - source_upper
        else:
            raise RuntimeError("ocra.source_anchor_mode must be 'absolute' or 'robot_shoulder_scaled'")

        source_points = np.vstack([source_shoulder, source_elbow, source_wrist])
        source_chain_length = (
            np.linalg.norm(source_elbow - source_shoulder)
            + np.linalg.norm(source_wrist - source_elbow)
        )
        normalizer = max(float(source_chain_length), 1e-6)
        sqrt_alpha = math.sqrt(alpha)
        sqrt_beta = math.sqrt(beta)
        target_position_weight = max(0.0, float(target_position_weight))
        joint_regularization_weight = max(0.0, float(joint_regularization_weight))

        def robot_skeleton(q_state):
            transform, joint_positions, _ = self.fk(q_state)
            tcp = transform[:3, 3]
            elbow = joint_positions[elbow_joint_index]
            if shoulder_joint_index is None:
                shoulder = robot_shoulder_anchor
            else:
                shoulder = joint_positions[int(shoulder_joint_index)]
            return np.vstack([shoulder, elbow, tcp]), transform

        def error_vector(q_state):
            robot_points, transform = robot_skeleton(q_state)
            tcp = transform[:3, 3]
            skeleton_error = ((source_points - robot_points) / normalizer).reshape(-1)
            orientation_error = orientation_error_vector(transform[:3, :3], desired_rotation)
            blocks = [
                sqrt_alpha * skeleton_error,
                sqrt_beta * orientation_error,
            ]
            if target_position_weight > 0.0:
                blocks.append(target_position_weight * ((target - tcp) / normalizer))
            if joint_regularization_weight > 0.0:
                blocks.append(joint_regularization_weight * (reference_q - q_state))
            return np.concatenate(blocks)

        def metrics(q_state):
            robot_points, transform = robot_skeleton(q_state)
            tcp = transform[:3, 3]
            skeleton_delta = source_points - robot_points
            per_point = np.linalg.norm(skeleton_delta, axis=1)
            orientation = orientation_error_vector(transform[:3, :3], desired_rotation)
            position = target - tcp
            objective = (
                alpha * float(np.dot((skeleton_delta / normalizer).reshape(-1), (skeleton_delta / normalizer).reshape(-1)))
                + beta * float(np.dot(orientation, orientation))
            )
            if target_position_weight > 0.0:
                objective += target_position_weight * float(np.dot(position / normalizer, position / normalizer))
            if joint_regularization_weight > 0.0:
                regularization = reference_q - q_state
                objective += joint_regularization_weight * float(np.dot(regularization, regularization))
            return {
                "ocra_alpha": alpha,
                "ocra_beta": beta,
                "ocra_objective": objective,
                "ocra_source_anchor_mode": source_anchor_mode,
                "ocra_skeleton_rmse_m": float(math.sqrt(np.mean(per_point * per_point))),
                "ocra_skeleton_max_m": float(np.max(per_point)),
                "ocra_skeleton_shoulder_error_m": float(per_point[0]),
                "ocra_skeleton_elbow_error_m": float(per_point[1]),
                "ocra_skeleton_wrist_error_m": float(per_point[2]),
                "ocra_orientation_norm_rad": float(np.linalg.norm(orientation)),
                "ocra_target_position_error_m": float(np.linalg.norm(position)),
                "ocra_source_shoulder_base": source_shoulder,
                "ocra_source_elbow_base": source_elbow,
                "ocra_source_wrist_base": source_wrist,
            }

        last_metrics = None
        for _ in range(iterations):
            error = error_vector(q)
            last_metrics = metrics(q)
            if (
                last_metrics["ocra_skeleton_rmse_m"] < skeleton_tolerance
                and last_metrics["ocra_orientation_norm_rad"] < orientation_tolerance
                and (
                    target_position_weight <= 0.0
                    or last_metrics["ocra_target_position_error_m"] < position_tolerance
                )
            ):
                return (
                    q,
                    True,
                    last_metrics["ocra_target_position_error_m"],
                    last_metrics["ocra_orientation_norm_rad"],
                    last_metrics["ocra_skeleton_rmse_m"],
                    last_metrics,
                )

            jacobian = np.zeros((len(error), joint_count), dtype=float)
            for idx in range(joint_count):
                q_eps = q.copy()
                q_eps[idx] += finite_difference_step
                jacobian[:, idx] = (error_vector(q_eps) - error) / finite_difference_step
            dq = -self.damped_pseudoinverse(jacobian, damping) @ error
            largest_step = float(np.max(np.abs(dq)))
            if largest_step > max_step:
                dq *= max_step / largest_step
            q += dq
            for idx, (lower, upper) in enumerate(self.limits):
                if math.isfinite(lower) or math.isfinite(upper):
                    q[idx] = np.clip(q[idx], lower, upper)

        last_metrics = metrics(q)
        return (
            q,
            False,
            last_metrics["ocra_target_position_error_m"],
            last_metrics["ocra_orientation_norm_rad"],
            last_metrics["ocra_skeleton_rmse_m"],
            last_metrics,
        )


def world_to_robot_base(point_world):
    c = math.cos(-ROBOT_WORLD_YAW)
    s = math.sin(-ROBOT_WORLD_YAW)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return rotation @ (np.asarray(point_world, dtype=float) - ROBOT_WORLD_POS)


def robot_base_to_world(point_base):
    c = math.cos(ROBOT_WORLD_YAW)
    s = math.sin(ROBOT_WORLD_YAW)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return ROBOT_WORLD_POS + rotation @ np.asarray(point_base, dtype=float)


def phase_target(t):
    ready = SMPLX_KEYPOINTS_WORLD["ready"]
    hover = SMPLX_KEYPOINTS_WORLD["hover"]
    grasp = SMPLX_KEYPOINTS_WORLD["grasp"]
    lift = SMPLX_KEYPOINTS_WORLD["lift"]
    if t < 1.0:
        return ready, "ready"
    if t < 4.8:
        alpha = smoothstep((t - 1.0) / 3.8)
        return ready * (1.0 - alpha) + hover * alpha, "over-table-approach"
    if t < 6.4:
        alpha = smoothstep((t - 4.8) / 1.6)
        return hover * (1.0 - alpha) + grasp * alpha, "descend"
    if t < 7.4:
        return grasp, "grasp"
    alpha = smoothstep((t - 7.4) / 2.6)
    return grasp * (1.0 - alpha) + lift * alpha, "lift"


def retarget_target(point_world):
    base_target = world_to_robot_base(point_world)
    return clip_base_target(base_target)


def clip_base_target(base_target):
    base_target = np.asarray(base_target, dtype=float).copy()
    limits = RETARGET_RUNTIME_CONFIG.get("target_limits_base", {})
    min_values = limits.get("min", [-0.72, -0.68, 0.1])
    max_values = limits.get("max", [0.45, 0.25, None])
    for idx in range(3):
        if min_values[idx] is not None:
            base_target[idx] = max(base_target[idx], float(min_values[idx]))
        if max_values[idx] is not None:
            base_target[idx] = min(base_target[idx], float(max_values[idx]))
    table_safety = RETARGET_RUNTIME_CONFIG.get("safety", {}).get("table", {})
    if table_safety.get("enabled", False):
        min_tcp_z = table_safety.get("min_tcp_z_base")
        if min_tcp_z is not None:
            base_target[2] = max(base_target[2], float(min_tcp_z))
    return base_target


def conditioned_base_target(base_target, t):
    result = np.asarray(base_target, dtype=float).copy()
    conditioning = RETARGET_RUNTIME_CONFIG.get("workspace_conditioning", {})
    result = global_workspace_transform_base(result)
    for window in conditioning.get("local_windows", []):
        center = float(window.get("center_time", window.get("center_seconds", 0.0)))
        half_width = max(1e-6, float(window.get("half_width_seconds", window.get("half_width", 0.0))))
        distance = abs(float(t) - center)
        if distance >= half_width:
            continue
        target_base = vector3(window.get("target_base", result.tolist()), "workspace_conditioning.local_windows.target_base")
        strength = float(window.get("strength", 1.0))
        u = 1.0 - distance / half_width
        if window.get("profile", "smoothstep") == "linear":
            alpha = u
        else:
            alpha = smoothstep(u)
        alpha = float(np.clip(strength * alpha, 0.0, 1.0))
        axes = window.get("axes", [0, 1, 2])
        for axis in axes:
            axis = int(axis)
            result[axis] = result[axis] * (1.0 - alpha) + target_base[axis] * alpha
    return result


def global_workspace_transform_base(base_point):
    result = np.asarray(base_point, dtype=float).copy()
    transform = RETARGET_RUNTIME_CONFIG.get("workspace_conditioning", {}).get("global_affine", {})
    if not transform or not transform.get("enabled", False):
        return result
    source_min = vector3(transform.get("source_min", [0.0, 0.0, 0.0]), "workspace_conditioning.global_affine.source_min")
    source_max = vector3(transform.get("source_max", [1.0, 1.0, 1.0]), "workspace_conditioning.global_affine.source_max")
    target_min = vector3(transform.get("target_min", source_min.tolist()), "workspace_conditioning.global_affine.target_min")
    target_max = vector3(transform.get("target_max", source_max.tolist()), "workspace_conditioning.global_affine.target_max")
    axes = transform.get("axes", [0, 1, 2])
    for axis in axes:
        axis = int(axis)
        denom = source_max[axis] - source_min[axis]
        if abs(float(denom)) < 1e-9:
            continue
        ratio = (result[axis] - source_min[axis]) / denom
        if transform.get("clip_ratio", True):
            ratio = float(np.clip(ratio, 0.0, 1.0))
        result[axis] = target_min[axis] + ratio * (target_max[axis] - target_min[axis])
    return result


def retarget_keypoints_base():
    return {name: retarget_target(point) for name, point in SMPLX_KEYPOINTS_WORLD.items()}


def cup_task_keypoints_base():
    if CUP_TASK_KEYPOINTS_BASE:
        return {name: point.copy() for name, point in CUP_TASK_KEYPOINTS_BASE.items()}
    return {name: retarget_target(point) for name, point in CUP_TASK_KEYPOINTS_WORLD.items()}


def downward_gripper_rotation_base():
    orientation = RETARGET_RUNTIME_CONFIG.get("orientation", {})
    rotation = np.asarray(
        orientation.get("rotation_base", [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        dtype=float,
    )
    if rotation.shape != (3, 3):
        raise RuntimeError("orientation.rotation_base must be a 3x3 matrix")
    return rotation


def orientation_error_vector(current_rotation, desired_rotation):
    current_rotation = np.asarray(current_rotation, dtype=float)
    desired_rotation = np.asarray(desired_rotation, dtype=float)
    return 0.5 * (
        np.cross(current_rotation[:, 0], desired_rotation[:, 0])
        + np.cross(current_rotation[:, 1], desired_rotation[:, 1])
        + np.cross(current_rotation[:, 2], desired_rotation[:, 2])
    )


def normalized(vector, fallback):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float)
    return vector / norm


def included_angle(vector_a, vector_b):
    unit_a = normalized(vector_a, [1.0, 0.0, 0.0])
    unit_b = normalized(vector_b, [1.0, 0.0, 0.0])
    return math.acos(float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)))


def forearm_aligned_rotation_base(wrist_base, elbow_base, shoulder_base=None):
    orientation = RETARGET_RUNTIME_CONFIG.get("orientation", {})
    forearm = np.asarray(wrist_base, dtype=float) - np.asarray(elbow_base, dtype=float)
    axis = orientation.get("forearm_axis", "z")
    z_axis = normalized(forearm, downward_gripper_rotation_base()[:, 2])
    if axis == "-z":
        z_axis = -z_axis
    elif axis != "z":
        raise RuntimeError("orientation.forearm_axis currently supports only 'z' or '-z'")

    if shoulder_base is not None:
        upper_arm = np.asarray(elbow_base, dtype=float) - np.asarray(shoulder_base, dtype=float)
        reference = normalized(upper_arm, orientation.get("reference_up_base", [0.0, 0.0, 1.0]))
    else:
        reference = normalized(orientation.get("reference_up_base", [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    x_axis = np.cross(reference, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.asarray([1.0, 0.0, 0.0]), z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.asarray([0.0, 1.0, 0.0]), z_axis)
    x_axis = normalized(x_axis, [1.0, 0.0, 0.0])
    y_axis = normalized(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    return np.column_stack([x_axis, y_axis, z_axis])


def table_edge_gripper_rotation_base(tcp_base):
    orientation = RETARGET_RUNTIME_CONFIG.get("orientation", {})
    tcp_base = np.asarray(tcp_base, dtype=float)
    fallback_target = clip_base_target(world_to_robot_base(TCP_GRASP_WORLD + np.array([0.0, 0.0, 0.08], dtype=float)))
    approach_target = vector3(
        orientation.get("approach_target_base", fallback_target.tolist()),
        "orientation.approach_target_base",
    )
    approach_direction = normalized(
        approach_target - tcp_base,
        orientation.get("approach_direction_base", [-0.75, -0.65, -0.15]),
    )
    downward_blend = float(orientation.get("downward_blend", 0.18))
    z_axis = normalized(
        (1.0 - downward_blend) * approach_direction + downward_blend * np.asarray([0.0, 0.0, -1.0], dtype=float),
        downward_gripper_rotation_base()[:, 2],
    )

    axis = orientation.get("approach_axis", orientation.get("forearm_axis", "z"))
    if axis == "-z":
        z_axis = -z_axis
    elif axis != "z":
        raise RuntimeError("orientation.approach_axis currently supports only 'z' or '-z'")

    reference = normalized(orientation.get("reference_up_base", [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    x_axis = np.cross(reference, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.asarray([1.0, 0.0, 0.0]), z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.asarray([0.0, 1.0, 0.0]), z_axis)
    x_axis = normalized(x_axis, [1.0, 0.0, 0.0])
    y_axis = normalized(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    return np.column_stack([x_axis, y_axis, z_axis])


def full_arm_elbow_target_base(wrist_base, shoulder_base, elbow_base):
    config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
    shoulder_anchor = vector3(
        config.get("robot_shoulder_anchor_base", [0.0, 0.0, 0.267]),
        "full_arm.robot_shoulder_anchor_base",
    )
    upper_length = float(config.get("robot_upper_arm_length", 0.30))
    min_angle = math.radians(float(config.get("min_bend_angle_deg", 20.0)))
    max_angle = math.radians(float(config.get("max_bend_angle_deg", 135.0)))

    wrist_base = np.asarray(wrist_base, dtype=float)
    shoulder_base = np.asarray(shoulder_base, dtype=float)
    elbow_base = np.asarray(elbow_base, dtype=float)

    human_line = normalized(wrist_base - shoulder_base, [1.0, 0.0, 0.0])
    human_upper = normalized(elbow_base - shoulder_base, human_line)
    bend = human_upper - np.dot(human_upper, human_line) * human_line
    robot_line = normalized(wrist_base - shoulder_anchor, [1.0, 0.0, 0.0])
    bend = bend - np.dot(bend, robot_line) * robot_line
    bend_dir = normalized(bend, np.cross([0.0, 0.0, 1.0], robot_line))
    if np.linalg.norm(bend_dir) < 1e-6:
        bend_dir = normalized(np.cross([0.0, 1.0, 0.0], robot_line), [0.0, 0.0, 1.0])

    bend_angle = math.acos(float(np.clip(np.dot(human_upper, human_line), -1.0, 1.0)))
    bend_angle = float(np.clip(bend_angle, min_angle, max_angle))
    elbow_dir = normalized(
        math.cos(bend_angle) * robot_line + math.sin(bend_angle) * bend_dir,
        robot_line,
    )
    return shoulder_anchor + upper_length * elbow_dir


def robot_shoulder_anchor_base():
    functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
    full_arm_config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
    return vector3(
        functional_config.get(
            "robot_shoulder_anchor_base",
            full_arm_config.get("robot_shoulder_anchor_base", [0.0, 0.0, 0.267]),
        ),
        "functional.robot_shoulder_anchor_base",
    )


def functional_source_vector_rotation_base():
    functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
    rotation = np.asarray(
        functional_config.get(
            "source_vector_rotation_base",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        dtype=float,
    )
    if rotation.shape != (3, 3):
        raise RuntimeError("functional.source_vector_rotation_base must be a 3x3 matrix")
    return rotation


def robot_elbow_joint_index():
    functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
    full_arm_config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
    return int(functional_config.get("robot_elbow_joint_index", full_arm_config.get("robot_elbow_joint_index", 3)))


def robot_shoulder_joint_index():
    functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
    index = functional_config.get("robot_shoulder_joint_index")
    if index is None:
        return None
    return int(index)


def anti_self_insertion_metrics_for_fk(tcp_base, joint_positions, rotation_base=None):
    config = RETARGET_RUNTIME_CONFIG.get("anti_self_insertion", {})
    if not config.get("enabled", False):
        return {
            "enabled": False,
            "penalty": 0.0,
            "forearm_alignment": 1.0,
            "tool_alignment": 1.0,
        }
    joint_positions = np.asarray(joint_positions, dtype=float)
    elbow_index = robot_elbow_joint_index()
    if elbow_index < 0 or elbow_index >= len(joint_positions):
        return {
            "enabled": True,
            "penalty": 0.0,
            "forearm_alignment": 1.0,
            "tool_alignment": 1.0,
        }
    tcp_base = np.asarray(tcp_base, dtype=float)
    elbow_base = joint_positions[elbow_index]
    safe_direction = normalized(config.get("safe_direction_base", [-0.75, -0.65, 0.0]), [-1.0, 0.0, 0.0])
    forearm_direction = normalized(tcp_base - elbow_base, safe_direction)
    forearm_alignment = float(np.dot(forearm_direction, safe_direction))
    min_alignment = float(config.get("min_forearm_alignment", -0.10))
    forearm_penalty = max(0.0, min_alignment - forearm_alignment)
    tool_alignment = 1.0
    tool_penalty = 0.0
    if rotation_base is not None:
        rotation_base = np.asarray(rotation_base, dtype=float)
        tool_axis = rotation_base[:, 2]
        if config.get("tool_axis", "z") == "-z":
            tool_axis = -tool_axis
        elif config.get("tool_axis", "z") != "z":
            raise RuntimeError("anti_self_insertion.tool_axis currently supports only 'z' or '-z'")
        tool_direction = normalized(tool_axis, safe_direction)
        tool_alignment = float(np.dot(tool_direction, safe_direction))
        min_tool_alignment = float(config.get("min_tool_alignment", 0.10))
        tool_penalty = max(0.0, min_tool_alignment - tool_alignment)
    penalty = forearm_penalty + float(config.get("tool_penalty_scale", 1.0)) * tool_penalty
    return {
        "enabled": True,
        "penalty": float(penalty),
        "forearm_penalty": float(forearm_penalty),
        "tool_penalty": float(tool_penalty),
        "forearm_alignment": forearm_alignment,
        "tool_alignment": tool_alignment,
        "min_forearm_alignment": min_alignment,
    }


def compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base):
    transform, joint_positions, _ = chain.fk(q)
    elbow_index = robot_elbow_joint_index()
    shoulder_index = robot_shoulder_joint_index()
    robot_elbow = joint_positions[elbow_index]
    robot_shoulder = joint_positions[shoulder_index] if shoulder_index is not None else robot_shoulder_anchor_base()
    metrics = functional_mapping_metrics(
        shoulder_base,
        elbow_base,
        wrist_base,
        robot_shoulder,
        robot_elbow,
        transform[:3, 3],
    )
    anti_metrics = anti_self_insertion_metrics_for_fk(transform[:3, 3], joint_positions, transform[:3, :3])
    metrics["anti_self_penalty"] = float(anti_metrics.get("penalty", 0.0))
    metrics["anti_self_tool_penalty"] = float(anti_metrics.get("tool_penalty", 0.0))
    metrics["anti_self_forearm_alignment"] = float(anti_metrics.get("forearm_alignment", 1.0))
    metrics["anti_self_tool_alignment"] = float(anti_metrics.get("tool_alignment", 1.0))
    return metrics


def functional_solution_score(error, functional_metrics, q, reference_q, position_tolerance, functional_config):
    if functional_metrics is None:
        return float("inf")
    position_score = float(error) / max(float(position_tolerance), 1e-6)
    forearm_score = float(functional_metrics.get("forearm_max_rad", 0.0))
    upper_score = float(functional_metrics.get("upper_arm_max_rad", 0.0))
    joint_score = 0.0
    if reference_q is not None:
        joint_score = float(np.linalg.norm(np.asarray(q, dtype=float) - np.asarray(reference_q, dtype=float)))
    return (
        float(functional_config.get("branch_recovery_position_weight", 2.0)) * position_score
        + float(functional_config.get("branch_recovery_forearm_weight", 1.0)) * forearm_score
        + float(functional_config.get("branch_recovery_upper_arm_weight", 1.0)) * upper_score
        + float(functional_config.get("branch_recovery_joint_weight", 0.08)) * joint_score
    )


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


def load_reconstructed_trajectory(path):
    global LOADED_TRAJECTORY_METADATA
    if path is None:
        LOADED_TRAJECTORY_METADATA = {}
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", "unknown")
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError(f"Reconstructed trajectory has no samples: {path}")
    LOADED_TRAJECTORY_METADATA = {
        "schema": schema,
        "source": payload.get("source", "unknown"),
        "path": str(path),
    }
    trajectory = []
    for sample in samples:
        if sample.get("confidence", 1.0) <= 0.0:
            continue
        if "right_wrist_world" in sample:
            world = sample["right_wrist_world"]
        elif "right_hand_world" in sample:
            world = sample["right_hand_world"]
        else:
            raise RuntimeError(
                f"Trajectory sample is missing right_wrist_world/right_hand_world for schema={schema}"
            )
        base = sample.get("right_wrist_base", sample.get("right_hand_base"))
        trajectory.append(
            {
                "time": float(sample["time"]),
                "phase": sample.get("phase", "reconstructed"),
                "world": np.asarray(world, dtype=float),
                "base": np.asarray(base, dtype=float) if base is not None else None,
                "right_shoulder_world": np.asarray(sample["right_shoulder_world"], dtype=float)
                if "right_shoulder_world" in sample
                else None,
                "right_elbow_world": np.asarray(sample["right_elbow_world"], dtype=float)
                if "right_elbow_world" in sample
                else None,
            }
        )
    if not trajectory:
        raise RuntimeError(f"Reconstructed trajectory has no confident samples: {path}")
    trajectory.sort(key=lambda item: item["time"])
    return trajectory


def interpolate_optional_vector(before, after, key, alpha):
    if before.get(key) is None or after.get(key) is None:
        return None
    return before[key] * (1.0 - alpha) + after[key] * alpha


def sample_reconstructed_world(trajectory, t):
    if t <= trajectory[0]["time"]:
        return trajectory[0]["world"].copy(), trajectory[0]["phase"]
    if t >= trajectory[-1]["time"]:
        return trajectory[-1]["world"].copy(), trajectory[-1]["phase"]
    for before, after in zip(trajectory[:-1], trajectory[1:]):
        if before["time"] <= t <= after["time"]:
            span = max(1e-6, after["time"] - before["time"])
            alpha = (t - before["time"]) / span
            world = before["world"] * (1.0 - alpha) + after["world"] * alpha
            return world, after["phase"]
    return trajectory[-1]["world"].copy(), trajectory[-1]["phase"]


def sample_reconstructed_target(trajectory, t):
    if t <= trajectory[0]["time"]:
        item = trajectory[0]
        base = None if item.get("base") is None else item["base"].copy()
        arm = {
            "right_shoulder_world": None
            if item.get("right_shoulder_world") is None
            else item["right_shoulder_world"].copy(),
            "right_elbow_world": None
            if item.get("right_elbow_world") is None
            else item["right_elbow_world"].copy(),
        }
        return item["world"].copy(), base, item["phase"], arm
    if t >= trajectory[-1]["time"]:
        item = trajectory[-1]
        base = None if item.get("base") is None else item["base"].copy()
        arm = {
            "right_shoulder_world": None
            if item.get("right_shoulder_world") is None
            else item["right_shoulder_world"].copy(),
            "right_elbow_world": None
            if item.get("right_elbow_world") is None
            else item["right_elbow_world"].copy(),
        }
        return item["world"].copy(), base, item["phase"], arm
    for before, after in zip(trajectory[:-1], trajectory[1:]):
        if before["time"] <= t <= after["time"]:
            span = max(1e-6, after["time"] - before["time"])
            alpha = (t - before["time"]) / span
            world = before["world"] * (1.0 - alpha) + after["world"] * alpha
            base = None
            if before.get("base") is not None and after.get("base") is not None:
                base = before["base"] * (1.0 - alpha) + after["base"] * alpha
            arm = {
                "right_shoulder_world": interpolate_optional_vector(before, after, "right_shoulder_world", alpha),
                "right_elbow_world": interpolate_optional_vector(before, after, "right_elbow_world", alpha),
            }
            return world, base, after["phase"], arm
    item = trajectory[-1]
    base = None if item.get("base") is None else item["base"].copy()
    arm = {
        "right_shoulder_world": None
        if item.get("right_shoulder_world") is None
        else item["right_shoulder_world"].copy(),
        "right_elbow_world": None
        if item.get("right_elbow_world") is None
        else item["right_elbow_world"].copy(),
    }
    return item["world"].copy(), base, item["phase"], arm


def loaded_trajectory_phase_prefix():
    schema = LOADED_TRAJECTORY_METADATA.get("schema", "")
    if schema.startswith("smplx_fit_right_arm_trajectory"):
        return "smplx-fit"
    if "d455" in schema:
        return "d455"
    return "trajectory"


def first_phase_time(trajectory, phase_name, fallback):
    for sample in trajectory:
        if sample["phase"] == phase_name:
            return sample["time"]
    return fallback


def cup_constrained_world_target(trajectory, t):
    descend_start = first_phase_time(trajectory, "descend", 4.8)
    grasp_start = first_phase_time(trajectory, "grasp", 6.4)
    lift_start = first_phase_time(trajectory, "lift", 7.4)
    final_time = max(trajectory[-1]["time"], lift_start + 2.0)
    ready_hold = min(1.2, max(0.0, descend_start - 0.4))

    ready = CUP_TASK_KEYPOINTS_WORLD["ready"]
    hover = CUP_TASK_KEYPOINTS_WORLD["hover"]
    grasp = CUP_TASK_KEYPOINTS_WORLD["grasp"]
    lift = CUP_TASK_KEYPOINTS_WORLD["lift"]

    if t < ready_hold:
        return ready, "cup-constrained-ready"
    if t < descend_start:
        alpha = smoothstep((t - ready_hold) / max(0.1, descend_start - ready_hold))
        return ready * (1.0 - alpha) + hover * alpha, "cup-constrained-approach"
    if t < grasp_start:
        alpha = smoothstep((t - descend_start) / max(0.1, grasp_start - descend_start))
        return hover * (1.0 - alpha) + grasp * alpha, "cup-constrained-descend"
    if t < lift_start:
        return grasp, "cup-constrained-grasp"
    alpha = smoothstep((t - lift_start) / max(0.1, final_time - lift_start))
    return grasp * (1.0 - alpha) + lift * alpha, "cup-constrained-lift"


def wait_for_joint_state(node, timeout=5.0):
    result = {"msg": None}

    def callback(msg):
        result["msg"] = msg

    subscription = node.create_subscription(JointState, "/joint_states", callback, 10)
    deadline = time.monotonic() + timeout
    while rclpy.ok() and result["msg"] is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(subscription)
    if result["msg"] is None:
        raise RuntimeError("Timed out waiting for /joint_states")
    by_name = dict(zip(result["msg"].name, result["msg"].position))
    return np.asarray([by_name[name] for name in JOINT_NAMES], dtype=float)


def get_robot_description(node):
    params = node.get_parameters_by_prefix("")
    if "robot_description" in params:
        return params["robot_description"].value
    client = node.create_client(rclpy.parameter_client.GetParameters, "/robot_state_publisher/get_parameters")
    raise RuntimeError("robot_description parameter lookup failed")


def fetch_robot_description_with_cli():
    import subprocess

    output = subprocess.check_output(
        ["ros2", "param", "get", "/robot_state_publisher", "robot_description"],
        text=True,
    )
    return clean_robot_description(output.strip())


def build_trajectory(chain, seed, seconds, fps, reconstructed_trajectory=None, cup_constrained=True):
    frame_count = int(round(seconds * fps))
    q = seed.copy()
    reference_q = seed.copy()
    start_tcp = chain.fk(seed)[0][:3, 3]
    ik_config = RETARGET_RUNTIME_CONFIG.get("ik", {})
    controller_blend_in = float(ik_config.get("controller_blend_in_seconds", 1.0))
    orientation_mode = RETARGET_RUNTIME_CONFIG.get("orientation", {}).get("mode", "tcp_z_down")
    default_desired_rotation = downward_gripper_rotation_base()
    phase_prefix = loaded_trajectory_phase_prefix()
    if reconstructed_trajectory is None:
        keypoints = retarget_keypoints_base()
        hover = keypoints["hover"]
        grasp = keypoints["grasp"]
        lift = keypoints["lift"]
    points = []
    diagnostics = []
    for frame in range(frame_count + 1):
        t = min(seconds, frame / fps)
        desired_rotation = default_desired_rotation
        full_arm_elbow_target = None
        functional_metrics = None
        wrist_base = None
        elbow_base = None
        shoulder_base = None
        if reconstructed_trajectory is not None:
            if cup_constrained:
                point_world, phase = cup_constrained_world_target(reconstructed_trajectory, t)
                observed_phase = phase
                observed_target = retarget_target(point_world)
            else:
                point_world, point_base, observed_phase, arm = sample_reconstructed_target(reconstructed_trajectory, t)
                raw_base_target = world_to_robot_base(point_world) if point_base is None else np.asarray(point_base, dtype=float)
                conditioned_target = conditioned_base_target(raw_base_target, t)
                observed_target = clip_base_target(conditioned_target)
                if orientation_mode in (
                    "forearm_aligned",
                    "full_arm_aligned",
                    "full_arm_hierarchical",
                    "functional_hierarchical",
                    "table_edge_soft_functional",
                    "ocra_baseline",
                ) and arm.get("right_elbow_world") is not None:
                    wrist_base = observed_target
                    elbow_base = global_workspace_transform_base(world_to_robot_base(arm["right_elbow_world"]))
                    shoulder_base = (
                        global_workspace_transform_base(world_to_robot_base(arm["right_shoulder_world"]))
                        if arm.get("right_shoulder_world") is not None
                        else None
                    )
                    if orientation_mode == "table_edge_soft_functional":
                        desired_rotation = table_edge_gripper_rotation_base(wrist_base)
                    else:
                        desired_rotation = forearm_aligned_rotation_base(wrist_base, elbow_base, shoulder_base)
                    if orientation_mode in ("full_arm_aligned", "full_arm_hierarchical") and shoulder_base is not None:
                        full_arm_elbow_target = full_arm_elbow_target_base(wrist_base, shoulder_base, elbow_base)
            if cup_constrained:
                if t < controller_blend_in:
                    alpha = smoothstep(t / controller_blend_in)
                    target = start_tcp * (1.0 - alpha) + observed_target * alpha
                else:
                    target = observed_target
                phase = f"{phase_prefix}-{observed_phase}"
            elif t < controller_blend_in:
                alpha = smoothstep(t / max(0.1, controller_blend_in))
                target = start_tcp * (1.0 - alpha) + observed_target * alpha
                phase = f"{phase_prefix}-{observed_phase}"
            else:
                target = observed_target
                phase = f"{phase_prefix}-{observed_phase}"
        else:
            if t < 4.8:
                alpha = smoothstep(t / 4.8)
                target = start_tcp * (1.0 - alpha) + hover * alpha
                phase = "workspace-entry"
            elif t < 6.4:
                alpha = smoothstep((t - 4.8) / 1.6)
                target = hover * (1.0 - alpha) + grasp * alpha
                phase = "descend"
            elif t < 7.4:
                target = grasp
                phase = "grasp"
            else:
                alpha = smoothstep((t - 7.4) / 2.6)
                target = grasp * (1.0 - alpha) + lift * alpha
                phase = "lift"
        use_pose_ik = (
            reconstructed_trajectory is not None
            and t >= controller_blend_in
            and orientation_mode != "free_wrist_position_only"
        )
        q_before_ik = q.copy()
        joint_delta_limited = False
        if use_pose_ik:
            if orientation_mode == "ocra_baseline" and shoulder_base is not None and elbow_base is not None:
                ocra_config = RETARGET_RUNTIME_CONFIG.get("ocra", {})
                q, converged, error, orientation_error, elbow_error, functional_metrics = chain.solve_ocra_baseline(
                    target,
                    desired_rotation,
                    wrist_base,
                    elbow_base,
                    shoulder_base,
                    q,
                    robot_shoulder_anchor=robot_shoulder_anchor_base(),
                    elbow_joint_index=int(ocra_config.get("robot_elbow_joint_index", robot_elbow_joint_index())),
                    shoulder_joint_index=ocra_config.get("robot_shoulder_joint_index", robot_shoulder_joint_index()),
                    iterations=int(ik_config.get("max_iterations", 180)),
                    damping=float(ocra_config.get("damping", 0.065)),
                    alpha=float(ocra_config.get("alpha", 0.67)),
                    beta=float(ocra_config.get("beta", 0.33)),
                    target_position_weight=float(ocra_config.get("target_position_weight", 0.0)),
                    joint_regularization_weight=float(ocra_config.get("joint_regularization_weight", 0.03)),
                    source_anchor_mode=ocra_config.get("source_anchor_mode", "absolute"),
                    robot_upper_arm_length=float(ocra_config.get("robot_upper_arm_length", 0.30)),
                    robot_forearm_length=float(ocra_config.get("robot_forearm_length", 0.34)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.025)),
                    skeleton_tolerance=float(ocra_config.get("skeleton_tolerance", 0.060)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.45)),
                    max_step=float(ocra_config.get("max_step", 0.060)),
                    finite_difference_step=float(ocra_config.get("finite_difference_step", 1e-4)),
                )
                solver_metrics = dict(functional_metrics or {})
                functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
                functional_metrics.update(solver_metrics)
                orientation_error = float(functional_metrics.get("ocra_orientation_norm_rad", orientation_error))
                elbow_error = float(functional_metrics.get("ocra_skeleton_rmse_m", elbow_error))
            elif orientation_mode == "table_edge_soft_functional" and shoulder_base is not None and elbow_base is not None:
                functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
                anti_config = RETARGET_RUNTIME_CONFIG.get("anti_self_insertion", {})
                q, converged, error, orientation_error, elbow_error, functional_metrics = chain.solve_table_edge_soft_functional_hierarchical(
                    target,
                    desired_rotation,
                    wrist_base,
                    elbow_base,
                    shoulder_base,
                    q,
                    robot_shoulder_anchor=robot_shoulder_anchor_base(),
                    elbow_joint_index=robot_elbow_joint_index(),
                    shoulder_joint_index=robot_shoulder_joint_index(),
                    iterations=int(ik_config.get("max_iterations", 120)),
                    primary_damping=float(functional_config.get("primary_damping", 0.045)),
                    secondary_damping=float(functional_config.get("secondary_damping", 0.075)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.014)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.30)),
                    forearm_tolerance=math.radians(float(functional_config.get("forearm_tolerance_deg", 70.0))),
                    upper_arm_tolerance=math.radians(float(functional_config.get("upper_arm_tolerance_deg", 85.0))),
                    anti_self_tolerance=float(anti_config.get("tolerance", 0.02)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.18)),
                    forearm_weight=float(functional_config.get("forearm_weight", 0.20)),
                    upper_arm_weight=float(functional_config.get("upper_arm_weight", 0.10)),
                    included_angle_weight=float(functional_config.get("included_angle_weight", 0.0)),
                    anti_self_weight=float(anti_config.get("weight", 0.70)),
                    joint_regularization_weight=float(functional_config.get("joint_regularization_weight", 0.06)),
                    secondary_step_scale=float(functional_config.get("secondary_step_scale", 0.65)),
                    max_step=float(functional_config.get("max_step", 0.055)),
                    finite_difference_step=float(functional_config.get("finite_difference_step", 1e-4)),
                    secondary_position_guard=float(functional_config.get("secondary_position_guard", 0.045)),
                )
                solver_metrics = dict(functional_metrics or {})
                functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
                transform_for_metrics = chain.fk(q)[0]
                orientation_error = float(
                    np.linalg.norm(orientation_error_vector(transform_for_metrics[:3, :3], desired_rotation))
                )
                functional_metrics["orientation_norm_rad"] = orientation_error
                functional_metrics["anti_self_penalty"] = float(
                    functional_metrics.get("anti_self_penalty", solver_metrics.get("anti_self_penalty", 0.0))
                )
                functional_metrics["anti_self_forearm_alignment"] = float(
                    functional_metrics.get(
                        "anti_self_forearm_alignment",
                        solver_metrics.get("anti_self_forearm_alignment", 1.0),
                    )
                )
                elbow_error = functional_metrics["upper_arm_max_rad"]
            elif orientation_mode == "functional_hierarchical" and shoulder_base is not None and elbow_base is not None:
                functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
                q, converged, error, orientation_error, elbow_error, functional_metrics = chain.solve_functional_hierarchical(
                    target,
                    wrist_base,
                    elbow_base,
                    shoulder_base,
                    q,
                    robot_shoulder_anchor=robot_shoulder_anchor_base(),
                    elbow_joint_index=robot_elbow_joint_index(),
                    shoulder_joint_index=robot_shoulder_joint_index(),
                    iterations=int(ik_config.get("max_iterations", 120)),
                    primary_damping=float(functional_config.get("primary_damping", 0.045)),
                    secondary_damping=float(functional_config.get("secondary_damping", 0.075)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.014)),
                    forearm_tolerance=math.radians(float(functional_config.get("forearm_tolerance_deg", 25.0))),
                    upper_arm_tolerance=math.radians(float(functional_config.get("upper_arm_tolerance_deg", 35.0))),
                    forearm_weight=float(functional_config.get("forearm_weight", 1.0)),
                    upper_arm_weight=float(functional_config.get("upper_arm_weight", 0.65)),
                    included_angle_weight=float(functional_config.get("included_angle_weight", 0.0)),
                    secondary_step_scale=float(functional_config.get("secondary_step_scale", 0.70)),
                    max_step=float(functional_config.get("max_step", 0.055)),
                    finite_difference_step=float(functional_config.get("finite_difference_step", 1e-4)),
                    secondary_position_guard=float(functional_config.get("secondary_position_guard", 0.035)),
                )
                functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
                orientation_error = functional_metrics["forearm_max_rad"]
                elbow_error = functional_metrics["upper_arm_max_rad"]
                if bool(functional_config.get("branch_recovery_enabled", True)):
                    position_tolerance = float(ik_config.get("position_tolerance", 0.014))
                    forearm_trigger = math.radians(float(functional_config.get("branch_recovery_forearm_trigger_deg", 65.0)))
                    upper_trigger = math.radians(float(functional_config.get("branch_recovery_upper_trigger_deg", 80.0)))
                    position_trigger = position_tolerance * float(functional_config.get("branch_recovery_position_trigger_scale", 2.5))
                    needs_recovery = (
                        functional_metrics["forearm_max_rad"] > forearm_trigger
                        or functional_metrics["upper_arm_max_rad"] > upper_trigger
                        or float(error) > position_trigger
                    )
                    if needs_recovery:
                        candidate_seeds = [reference_q.copy()]
                        recovery_joints = functional_config.get("branch_recovery_joints")
                        if recovery_joints is None:
                            recovery_joints = [
                                idx
                                for idx in (robot_shoulder_joint_index(), robot_elbow_joint_index())
                                if idx is not None
                            ]
                        recovery_offsets = functional_config.get(
                            "branch_recovery_offsets_rad",
                            [-1.2, -0.6, 0.6, 1.2],
                        )
                        for base_seed in (q_before_ik, reference_q):
                            for joint_idx in recovery_joints:
                                joint_idx = int(joint_idx)
                                if joint_idx < 0 or joint_idx >= len(base_seed):
                                    continue
                                for offset in recovery_offsets:
                                    candidate = np.asarray(base_seed, dtype=float).copy()
                                    candidate[joint_idx] += float(offset)
                                    lower, upper = chain.limits[joint_idx]
                                    if math.isfinite(lower) or math.isfinite(upper):
                                        candidate[joint_idx] = np.clip(candidate[joint_idx], lower, upper)
                                    candidate_seeds.append(candidate)
                        best = (
                            functional_solution_score(
                                error,
                                functional_metrics,
                                q,
                                q_before_ik,
                                position_tolerance,
                                functional_config,
                            ),
                            q,
                            converged,
                            error,
                            orientation_error,
                            elbow_error,
                            functional_metrics,
                        )
                        for candidate_seed in candidate_seeds:
                            cq, cconverged, cerror, _, _, _ = chain.solve_functional_hierarchical(
                                target,
                                wrist_base,
                                elbow_base,
                                shoulder_base,
                                candidate_seed,
                                robot_shoulder_anchor=robot_shoulder_anchor_base(),
                                elbow_joint_index=robot_elbow_joint_index(),
                                shoulder_joint_index=robot_shoulder_joint_index(),
                                iterations=int(ik_config.get("max_iterations", 120)),
                                primary_damping=float(functional_config.get("primary_damping", 0.045)),
                                secondary_damping=float(functional_config.get("secondary_damping", 0.075)),
                                position_tolerance=position_tolerance,
                                forearm_tolerance=math.radians(float(functional_config.get("forearm_tolerance_deg", 25.0))),
                                upper_arm_tolerance=math.radians(float(functional_config.get("upper_arm_tolerance_deg", 35.0))),
                                forearm_weight=float(functional_config.get("forearm_weight", 1.0)),
                                upper_arm_weight=float(functional_config.get("upper_arm_weight", 0.65)),
                                included_angle_weight=float(functional_config.get("included_angle_weight", 0.0)),
                                secondary_step_scale=float(functional_config.get("secondary_step_scale", 0.70)),
                                max_step=float(functional_config.get("max_step", 0.055)),
                                finite_difference_step=float(functional_config.get("finite_difference_step", 1e-4)),
                                secondary_position_guard=float(functional_config.get("secondary_position_guard", 0.035)),
                            )
                            cmetrics = compute_functional_metrics(chain, cq, wrist_base, elbow_base, shoulder_base)
                            corientation = cmetrics["forearm_max_rad"]
                            celbow = cmetrics["upper_arm_max_rad"]
                            cscore = functional_solution_score(
                                cerror,
                                cmetrics,
                                cq,
                                q_before_ik,
                                position_tolerance,
                                functional_config,
                            )
                            if cscore < best[0]:
                                best = (cscore, cq, cconverged, cerror, corientation, celbow, cmetrics)
                        _, q, converged, error, orientation_error, elbow_error, functional_metrics = best
            elif orientation_mode == "full_arm_hierarchical" and full_arm_elbow_target is not None:
                full_arm_config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
                q, converged, error, orientation_error, elbow_error = chain.solve_full_arm_hierarchical(
                    target,
                    desired_rotation,
                    full_arm_elbow_target,
                    q,
                    elbow_joint_index=int(full_arm_config.get("robot_elbow_joint_index", 3)),
                    iterations=int(ik_config.get("max_iterations", 80)),
                    primary_damping=float(full_arm_config.get("primary_damping", 0.040)),
                    secondary_damping=float(full_arm_config.get("secondary_damping", 0.070)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                    elbow_tolerance=float(full_arm_config.get("elbow_tolerance", 0.055)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
                    elbow_weight=float(full_arm_config.get("elbow_weight", 0.30)),
                    secondary_step_scale=float(full_arm_config.get("secondary_step_scale", 1.0)),
                    max_step=float(full_arm_config.get("max_step", 0.055)),
                )
            elif orientation_mode == "full_arm_aligned" and full_arm_elbow_target is not None:
                full_arm_config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
                q, converged, error, orientation_error, elbow_error = chain.solve_full_arm(
                    target,
                    desired_rotation,
                    full_arm_elbow_target,
                    q,
                    elbow_joint_index=int(full_arm_config.get("robot_elbow_joint_index", 3)),
                    iterations=int(ik_config.get("max_iterations", 80)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                    elbow_tolerance=float(full_arm_config.get("elbow_tolerance", 0.055)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
                    elbow_weight=float(full_arm_config.get("elbow_weight", 0.30)),
                )
            else:
                q, converged, error, orientation_error = chain.solve_pose(
                    target,
                    desired_rotation,
                    q,
                    iterations=int(ik_config.get("max_iterations", 80)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
                )
                elbow_error = None
        else:
            q, converged, error = chain.solve_position(target, q)
            orientation_error = None
            elbow_error = None
        max_frame_delta = ik_config.get("max_frame_joint_delta_rad")
        if frame > 0 and max_frame_delta is not None:
            max_frame_delta = float(max_frame_delta)
            delta = q - q_before_ik
            largest_delta = float(np.max(np.abs(delta)))
            if largest_delta > max_frame_delta:
                q = q_before_ik + delta * (max_frame_delta / largest_delta)
                joint_delta_limited = True
                converged = False
        if frame == 0:
            # The trajectory controller expects the first point to match the
            # current state. Keep the visual retarget target in diagnostics,
            # but send the exact seed as point 0 to avoid a path-tolerance
            # abort before the visible approach begins.
            q = seed.copy()
            transform = chain.fk(q)[0]
            error = float(np.linalg.norm(np.asarray(target, dtype=float) - transform[:3, 3]))
            converged = error < 0.006
            if full_arm_elbow_target is not None:
                _, joint_positions, _ = chain.fk(q)
                elbow_index = int(RETARGET_RUNTIME_CONFIG.get("full_arm", {}).get("robot_elbow_joint_index", 3))
                elbow_error = float(np.linalg.norm(full_arm_elbow_target - joint_positions[elbow_index]))
        else:
            transform, _, _ = chain.fk(q)
            if joint_delta_limited:
                error = float(np.linalg.norm(np.asarray(target, dtype=float) - transform[:3, 3]))
        if wrist_base is not None and elbow_base is not None and shoulder_base is not None:
            solver_metric_overrides = {
                key: value
                for key, value in (functional_metrics or {}).items()
                if key.startswith("ocra_")
            }
            functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
            functional_metrics.update(solver_metric_overrides)
            if orientation_mode == "table_edge_soft_functional":
                orientation_error = float(np.linalg.norm(orientation_error_vector(transform[:3, :3], desired_rotation)))
                functional_metrics["orientation_norm_rad"] = orientation_error
                elbow_error = functional_metrics["upper_arm_max_rad"]
            elif orientation_mode == "functional_hierarchical":
                orientation_error = functional_metrics["forearm_max_rad"]
                elbow_error = functional_metrics["upper_arm_max_rad"]
            elif orientation_mode == "ocra_baseline":
                orientation_error = float(functional_metrics.get("ocra_orientation_norm_rad", orientation_error or 0.0))
                elbow_error = float(functional_metrics.get("ocra_skeleton_rmse_m", elbow_error or 0.0))
        diagnostics.append(
            {
                "time": round(t, 4),
                "phase": phase,
                "target": target.tolist(),
                "tcp": transform[:3, 3].tolist(),
                "error": error,
                "orientation_error": orientation_error,
                "elbow_error": elbow_error,
                "functional_metrics": functional_metrics,
                "joint_delta_limited": joint_delta_limited,
                "converged": converged,
            }
        )
        points.append(q.copy())
    return np.asarray(points), diagnostics


def send_trajectory(node, positions, seconds, fps, wait=True):
    client = ActionClient(node, FollowJointTrajectory, "/xarm7_traj_controller/follow_joint_trajectory")
    if not client.wait_for_server(timeout_sec=8.0):
        raise RuntimeError("xarm7 trajectory action server unavailable")
    execution_config = RETARGET_RUNTIME_CONFIG.get("execution", {})
    time_scale = max(0.1, float(execution_config.get("time_scale", 1.0)))
    path_tolerance = execution_config.get("path_tolerance_rad")
    goal_tolerance = execution_config.get("goal_tolerance_rad")
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    for idx, q in enumerate(positions):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        t = min(seconds, idx / fps) * time_scale
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        goal.trajectory.points.append(point)
    goal.goal_time_tolerance.sec = max(3, int(math.ceil(3 * time_scale)))
    if path_tolerance is not None:
        for name in JOINT_NAMES:
            tolerance = JointTolerance()
            tolerance.name = name
            tolerance.position = float(path_tolerance)
            goal.path_tolerance.append(tolerance)
    if goal_tolerance is not None:
        for name in JOINT_NAMES:
            tolerance = JointTolerance()
            tolerance.name = name
            tolerance.position = float(goal_tolerance)
            goal.goal_tolerance.append(tolerance)
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if not handle or not handle.accepted:
        raise RuntimeError("xarm7 trajectory goal rejected")
    accepted_file = os.environ.get("RETARGET_GOAL_ACCEPTED_FILE")
    if accepted_file:
        Path(accepted_file).write_text(
            json.dumps(
                {
                    "event": "trajectory_goal_accepted",
                    "monotonic_time": time.monotonic(),
                    "trajectory_seconds": float(seconds),
                    "time_scale": float(time_scale),
                    "points": len(goal.trajectory.points),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if wait:
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result().result
        return int(result.error_code)
    return 0


def moveit2_enabled():
    return bool(RETARGET_RUNTIME_CONFIG.get("moveit2", {}).get("enabled", False))


def apply_moveit_table_scene(node):
    moveit_config = RETARGET_RUNTIME_CONFIG.get("moveit2", {})
    table_config = RETARGET_RUNTIME_CONFIG.get("safety", {}).get("table", {})
    if not moveit_config.get("enabled", False) or not table_config.get("enabled", False):
        return False
    try:
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from shape_msgs.msg import SolidPrimitive
    except Exception as exc:
        raise RuntimeError(f"MoveIt2 messages are unavailable: {exc}") from exc

    service_name = moveit_config.get("apply_scene_service", "/apply_planning_scene")
    client = node.create_client(ApplyPlanningScene, service_name)
    if not client.wait_for_service(timeout_sec=float(moveit_config.get("service_timeout", 8.0))):
        raise RuntimeError(f"MoveIt2 apply planning scene service unavailable: {service_name}")

    box_config = table_config.get("collision_box", {})
    center = vector3(box_config.get("center_base", [-0.34, -0.20, -0.04]), "safety.table.collision_box.center_base")
    size = vector3(box_config.get("size", [0.95, 1.25, 0.08]), "safety.table.collision_box.size")
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [float(v) for v in size]
    pose = Pose()
    pose.position.x = float(center[0])
    pose.position.y = float(center[1])
    pose.position.z = float(center[2])
    pose.orientation.w = 1.0

    obj = CollisionObject()
    obj.header.frame_id = table_config.get("frame", BASE_LINK)
    obj.id = box_config.get("id", "gazebo_table_top")
    obj.primitives = [primitive]
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD

    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = [obj]

    req = ApplyPlanningScene.Request()
    req.scene = scene
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=float(moveit_config.get("service_timeout", 8.0)))
    result = future.result()
    if result is None or not result.success:
        raise RuntimeError("MoveIt2 failed to apply table collision object to the planning scene")
    print(
        "moveit_scene table_collision=applied "
        f"frame={obj.header.frame_id} center={np.round(center, 3)} size={np.round(size, 3)}"
    )
    return True


def plan_joint_path_moveit(node, start_q, goal_q, allowed_time=None):
    moveit_config = RETARGET_RUNTIME_CONFIG.get("moveit2", {})
    try:
        from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, RobotState
        from moveit_msgs.srv import GetMotionPlan
        from sensor_msgs.msg import JointState
    except Exception as exc:
        raise RuntimeError(f"MoveIt2 planning messages are unavailable: {exc}") from exc

    service_name = moveit_config.get("planning_service", "/plan_kinematic_path")
    client = node.create_client(GetMotionPlan, service_name)
    if not client.wait_for_service(timeout_sec=float(moveit_config.get("service_timeout", 8.0))):
        raise RuntimeError(f"MoveIt2 planning service unavailable: {service_name}")

    start_state = RobotState()
    start_state.joint_state = JointState()
    start_state.joint_state.name = JOINT_NAMES
    start_state.joint_state.position = [float(v) for v in start_q]

    constraints = Constraints()
    tolerance = float(moveit_config.get("joint_goal_tolerance", 0.018))
    for name, value in zip(JOINT_NAMES, goal_q):
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = float(value)
        constraint.tolerance_above = tolerance
        constraint.tolerance_below = tolerance
        constraint.weight = 1.0
        constraints.joint_constraints.append(constraint)

    request = MotionPlanRequest()
    request.group_name = moveit_config.get("group_name", "xarm7")
    request.pipeline_id = moveit_config.get("pipeline_id", "ompl")
    request.planner_id = moveit_config.get("planner_id", "RRTConnectkConfigDefault")
    request.num_planning_attempts = int(moveit_config.get("num_planning_attempts", 5))
    request.allowed_planning_time = float(allowed_time or moveit_config.get("allowed_planning_time", 4.0))
    request.max_velocity_scaling_factor = float(moveit_config.get("max_velocity_scaling_factor", 0.25))
    request.max_acceleration_scaling_factor = float(moveit_config.get("max_acceleration_scaling_factor", 0.25))
    request.start_state = start_state
    request.goal_constraints = [constraints]

    req = GetMotionPlan.Request()
    req.motion_plan_request = request
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=request.allowed_planning_time + 8.0)
    result = future.result()
    if result is None:
        raise RuntimeError("MoveIt2 planning service returned no result")
    response = result.motion_plan_response
    code = int(response.error_code.val)
    if code != 1:
        raise RuntimeError(f"MoveIt2 planning failed error_code={code}")
    return response.trajectory.joint_trajectory, float(response.planning_time)


def send_joint_trajectory_message(node, joint_trajectory, duration_hint=None):
    client = ActionClient(node, FollowJointTrajectory, "/xarm7_traj_controller/follow_joint_trajectory")
    if not client.wait_for_server(timeout_sec=8.0):
        raise RuntimeError("xarm7 trajectory action server unavailable")

    name_to_index = {name: idx for idx, name in enumerate(joint_trajectory.joint_names)}
    if any(name not in name_to_index for name in JOINT_NAMES):
        raise RuntimeError(f"MoveIt2 trajectory joint names do not cover xArm7 joints: {joint_trajectory.joint_names}")
    raw_points = list(joint_trajectory.points)
    if not raw_points:
        raise RuntimeError("MoveIt2 returned an empty joint trajectory")

    last_time = raw_points[-1].time_from_start.sec + raw_points[-1].time_from_start.nanosec * 1e-9
    scale = 1.0
    if duration_hint and last_time > 1e-6 and last_time < duration_hint:
        scale = float(duration_hint) / last_time

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    previous_time = 0.0
    for idx, raw_point in enumerate(raw_points):
        point = JointTrajectoryPoint()
        point.positions = [float(raw_point.positions[name_to_index[name]]) for name in JOINT_NAMES]
        t = raw_point.time_from_start.sec + raw_point.time_from_start.nanosec * 1e-9
        if idx == 0 and t <= 1e-6:
            t = 0.05
        t *= scale
        if t <= previous_time:
            t = previous_time + 0.05
        previous_time = t
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        goal.trajectory.points.append(point)
    goal.goal_time_tolerance.sec = 3

    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if not handle or not handle.accepted:
        raise RuntimeError("xarm7 MoveIt2 trajectory goal rejected")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    return int(result_future.result().result.error_code)


def send_moveit_planned_goal(node, start_q, goal_q, duration_hint=None, label="moveit-segment"):
    joint_trajectory, planning_time = plan_joint_path_moveit(node, start_q, goal_q)
    code = send_joint_trajectory_message(node, joint_trajectory, duration_hint=duration_hint)
    print(
        f"moveit_plan {label} result={code} "
        f"points={len(joint_trajectory.points)} planning_time={planning_time:.3f}"
    )
    return code


def send_moveit_chunked_trajectory(node, positions, seconds, fps):
    apply_moveit_table_scene(node)
    moveit_config = RETARGET_RUNTIME_CONFIG.get("moveit2", {})
    chunk_seconds = float(moveit_config.get("chunk_seconds", 1.0))
    stride = max(1, int(round(chunk_seconds * fps)))
    indices = list(range(stride, len(positions), stride))
    if not indices or indices[-1] != len(positions) - 1:
        indices.append(len(positions) - 1)
    codes = []
    for count, idx in enumerate(indices, start=1):
        start_q = wait_for_joint_state(node, timeout=2.0)
        goal_q = positions[idx]
        duration_hint = min(chunk_seconds, max(0.2, seconds - idx / max(1e-6, fps)))
        code = send_moveit_planned_goal(node, start_q, goal_q, duration_hint=duration_hint, label=f"chunk-{count:02d}@{idx/fps:.2f}s")
        codes.append(code)
        time.sleep(0.05)
    return 0 if all(code in (0, -5) for code in codes) else (codes[-1] if codes else 0)


def build_cartesian_segment(chain, seed, target, duration, fps, desired_rotation, phase):
    ik_config = RETARGET_RUNTIME_CONFIG.get("ik", {})
    frame_count = max(1, int(round(duration * fps)))
    q = np.asarray(seed, dtype=float).copy()
    start_tcp = chain.fk(q)[0][:3, 3]
    target = np.asarray(target, dtype=float)
    points = []
    diagnostics = []
    for frame in range(frame_count + 1):
        alpha = smoothstep(frame / frame_count)
        step_target = start_tcp * (1.0 - alpha) + target * alpha
        if frame == 0:
            transform = chain.fk(q)[0]
            error = float(np.linalg.norm(step_target - transform[:3, 3]))
            orientation_error = None
            converged = error < 0.006
        else:
            q, converged, error, orientation_error = chain.solve_pose(
                step_target,
                desired_rotation,
                q,
                iterations=int(ik_config.get("max_iterations", 80)),
                position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
            )
            transform = chain.fk(q)[0]
        diagnostics.append(
            {
                "phase": phase,
                "time": round(frame / fps, 4),
                "target": step_target.tolist(),
                "tcp": transform[:3, 3].tolist(),
                "error": error,
                "orientation_error": orientation_error,
                "converged": converged,
            }
        )
        points.append(q.copy())
    return np.asarray(points), diagnostics


def send_segmented_trajectory(node, chain, seed, fps):
    keypoints = cup_task_keypoints_base()
    desired_rotation = downward_gripper_rotation_base()
    segments = RETARGET_RUNTIME_CONFIG.get("execution", {}).get("segments", [])
    if not segments:
        segments = [
            {"name": "settle-ready", "to": "ready", "duration_seconds": 1.0},
            {"name": "approach-hover", "to": "hover", "duration_seconds": 3.5},
            {"name": "descend-grasp", "to": "grasp", "duration_seconds": 2.0},
            {"name": "hold-grasp", "to": "grasp", "duration_seconds": 0.7},
            {"name": "vertical-lift", "to": "lift", "duration_seconds": 2.5},
        ]

    codes = []
    q = np.asarray(seed, dtype=float).copy()
    if moveit2_enabled():
        apply_moveit_table_scene(node)
    for segment in segments:
        name = segment["name"]
        waypoint_name = segment["to"]
        if waypoint_name not in keypoints:
            raise RuntimeError(f"Segment {name} references unknown waypoint: {waypoint_name}")
        duration = float(segment.get("duration_seconds", segment.get("duration", 2.0)))
        target = keypoints[waypoint_name]
        positions, diagnostics = build_cartesian_segment(chain, q, target, duration, fps, desired_rotation, name)
        if moveit2_enabled() and segment.get("moveit2", True):
            code = send_moveit_planned_goal(node, q, positions[-1], duration_hint=duration, label=name)
        elif segment.get("path_mode") == "dense":
            code = send_trajectory(node, positions, duration, fps)
        else:
            code = send_position_goal(node, positions[-1], duration)
        try:
            q = wait_for_joint_state(node, timeout=2.0)
        except RuntimeError:
            q = positions[-1]
        final_tcp = chain.fk(q)[0][:3, 3]
        final_error = float(np.linalg.norm(target - final_tcp))
        max_error = max(item["error"] for item in diagnostics)
        print(
            f"segment {name} to={waypoint_name} result={code} duration={duration:.2f} "
            f"target={np.round(target, 3)} tcp={np.round(final_tcp, 3)} "
            f"err={final_error:.4f} planned_max_err={max_error:.4f}"
        )
        codes.append(code)
        time.sleep(float(segment.get("settle_seconds", 0.15)))
    return 0 if all(code in (0, -5) for code in codes) else (codes[-1] if codes else 0)


def send_position_goal(node, q, duration):
    client = ActionClient(node, FollowJointTrajectory, "/xarm7_traj_controller/follow_joint_trajectory")
    if not client.wait_for_server(timeout_sec=8.0):
        raise RuntimeError("xarm7 trajectory action server unavailable")
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in q]
    point.velocities = [0.0] * len(JOINT_NAMES)
    point.time_from_start.sec = int(duration)
    point.time_from_start.nanosec = int(round((duration - int(duration)) * 1e9))
    goal.trajectory.points = [point]
    goal.goal_time_tolerance.sec = 3
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if not handle or not handle.accepted:
        raise RuntimeError("xarm7 trajectory keyframe goal rejected")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    return int(result_future.result().result.error_code)


def send_keyframe_trajectory(node, positions, fps):
    milestone_config = RETARGET_RUNTIME_CONFIG.get("execution", {}).get("milestones", [])
    if milestone_config:
        milestones = []
        for item in milestone_config:
            time_seconds = item.get("time_seconds", item.get("time", "end"))
            if time_seconds == "end":
                idx = len(positions) - 1
            else:
                idx = int(round(float(time_seconds) * fps))
                idx = max(0, min(len(positions) - 1, idx))
            milestones.append((idx, float(item.get("duration_seconds", item.get("duration", 4.0))), item["name"]))
    else:
        milestones = [
            (int(round(4.8 * fps)), 6.0, "hover-over-cup"),
            (int(round(7.4 * fps)), 5.0, "descend-to-cup"),
            (int(round(8.0 * fps)), 2.0, "grasp-hold"),
            (len(positions) - 1, 6.0, "vertical-lift"),
        ]
    codes = []
    for idx, duration, name in milestones:
        code = send_position_goal(node, positions[idx], duration)
        print(f"keyframe {name} result={code}")
        codes.append(code)
        time.sleep(0.2)
    return codes[-1] if codes else 0


def main():
    parser = argparse.ArgumentParser(
        description="Retarget the SMPL-X right-hand grasp path to an xArm7 joint trajectory."
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keyframes", action="store_true", help="Execute sparse phase keyframes instead of one dense action.")
    parser.add_argument("--segmented", action="store_true", help="Execute cup-relative task waypoints as separate feedback segments.")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--prepose-only", action="store_true", help="Move to one sampled retarget pose and exit.")
    parser.add_argument("--prepose-time", type=float, default=0.0, help="Source trajectory time for --prepose-only.")
    parser.add_argument("--prepose-duration", type=float, default=3.0, help="Execution duration for --prepose-only.")
    parser.add_argument("--save-joint-trajectory", help="Write the solved xArm7 joint trajectory to JSON before execution.")
    parser.add_argument("--load-joint-trajectory", help="Replay a previously solved xArm7 joint trajectory JSON.")
    parser.add_argument(
        "--trajectory-json",
        help="Right-hand/right-arm trajectory JSON. Supports D455 reconstructed JSON and fitted SMPL-X right-arm JSON.",
    )
    parser.add_argument(
        "--retarget-config",
        help="Calibration/retarget JSON for robot pose, cup anchor, task keypoints, IK, and keyframe timing.",
    )
    parser.add_argument(
        "--raw-reconstructed",
        action="store_true",
        help="Follow reconstructed right_hand_world directly. Default keeps reconstructed phase timing but constrains the grasp to the cup.",
    )
    args = parser.parse_args()

    retarget_config = load_retarget_config(args.retarget_config)
    apply_retarget_config(retarget_config)

    rclpy.init()
    node = rclpy.create_node("smplx_to_xarm7_retarget")
    try:
        current_q = wait_for_joint_state(node)
        if args.load_joint_trajectory:
            with open(args.load_joint_trajectory, encoding="utf-8") as handle:
                payload = json.load(handle)
            positions = np.asarray(payload["positions"], dtype=float)
            seconds = float(payload.get("seconds", args.seconds))
            fps = float(payload.get("fps", args.fps))
            if len(positions):
                positions[0] = np.asarray(current_q, dtype=float)
            print(
                f"joint_trajectory_source={args.load_joint_trajectory} "
                f"points={len(positions)} seconds={seconds:.3f} fps={fps:.3f}"
            )
            if args.dry_run:
                return 0
            code = send_trajectory(node, positions, seconds, fps, wait=not args.no_wait)
            print(f"trajectory_result={code}")
            return 0 if code in (0, -5) else 5
        urdf_text = fetch_robot_description_with_cli()
        chain = KinematicChain(urdf_text, base_link=BASE_LINK, tip_link=TIP_LINK)
        reconstructed = load_reconstructed_trajectory(args.trajectory_json)
        print(
            f"retarget_config={args.retarget_config or 'builtin_defaults'} "
            f"schema={retarget_config.get('schema', 'unknown')} "
            f"base={BASE_LINK} tip={TIP_LINK}"
        )
        if reconstructed is not None:
            schema = LOADED_TRAJECTORY_METADATA.get("schema", "unknown")
            source = LOADED_TRAJECTORY_METADATA.get("source", "unknown")
            print(f"trajectory_source={schema} source={source} samples={len(reconstructed)} path={args.trajectory_json}")
            print(f"trajectory_alignment={'raw_reconstructed' if args.raw_reconstructed else 'cup_constrained_grasp_anchor'}")
        else:
            print("trajectory_source=builtin_smplx_keypoints")
        if args.prepose_only:
            if reconstructed is None:
                raise RuntimeError("--prepose-only requires --trajectory-json")
            point_world, point_base, observed_phase, arm = sample_reconstructed_target(reconstructed, args.prepose_time)
            raw_base_target = world_to_robot_base(point_world) if point_base is None else np.asarray(point_base, dtype=float)
            target = clip_base_target(conditioned_base_target(raw_base_target, args.prepose_time))
            orientation_mode = RETARGET_RUNTIME_CONFIG.get("orientation", {}).get("mode", "tcp_z_down")
            desired_rotation = downward_gripper_rotation_base()
            full_arm_elbow_target = None
            functional_metrics = None
            wrist_base = None
            elbow_base = None
            shoulder_base = None
            if orientation_mode in (
                "forearm_aligned",
                "full_arm_aligned",
                "full_arm_hierarchical",
                "functional_hierarchical",
                "table_edge_soft_functional",
                "ocra_baseline",
            ) and arm.get("right_elbow_world") is not None:
                wrist_base = target
                elbow_base = global_workspace_transform_base(world_to_robot_base(arm["right_elbow_world"]))
                shoulder_base = (
                    global_workspace_transform_base(world_to_robot_base(arm["right_shoulder_world"]))
                    if arm.get("right_shoulder_world") is not None
                    else None
                )
                if orientation_mode == "table_edge_soft_functional":
                    desired_rotation = table_edge_gripper_rotation_base(wrist_base)
                else:
                    desired_rotation = forearm_aligned_rotation_base(wrist_base, elbow_base, shoulder_base)
                if orientation_mode in ("full_arm_aligned", "full_arm_hierarchical") and shoulder_base is not None:
                    full_arm_elbow_target = full_arm_elbow_target_base(wrist_base, shoulder_base, elbow_base)
            ik_config = RETARGET_RUNTIME_CONFIG.get("ik", {})
            if orientation_mode == "free_wrist_position_only":
                q, converged, error = chain.solve_position(target, current_q)
                orientation_error = None
                elbow_error = None
            elif orientation_mode == "ocra_baseline" and shoulder_base is not None and elbow_base is not None:
                ocra_config = RETARGET_RUNTIME_CONFIG.get("ocra", {})
                q, converged, error, orientation_error, elbow_error, functional_metrics = chain.solve_ocra_baseline(
                    target,
                    desired_rotation,
                    wrist_base,
                    elbow_base,
                    shoulder_base,
                    current_q,
                    robot_shoulder_anchor=robot_shoulder_anchor_base(),
                    elbow_joint_index=int(ocra_config.get("robot_elbow_joint_index", robot_elbow_joint_index())),
                    shoulder_joint_index=ocra_config.get("robot_shoulder_joint_index", robot_shoulder_joint_index()),
                    iterations=int(ik_config.get("max_iterations", 180)),
                    damping=float(ocra_config.get("damping", 0.065)),
                    alpha=float(ocra_config.get("alpha", 0.67)),
                    beta=float(ocra_config.get("beta", 0.33)),
                    target_position_weight=float(ocra_config.get("target_position_weight", 0.0)),
                    joint_regularization_weight=float(ocra_config.get("joint_regularization_weight", 0.03)),
                    source_anchor_mode=ocra_config.get("source_anchor_mode", "absolute"),
                    robot_upper_arm_length=float(ocra_config.get("robot_upper_arm_length", 0.30)),
                    robot_forearm_length=float(ocra_config.get("robot_forearm_length", 0.34)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.025)),
                    skeleton_tolerance=float(ocra_config.get("skeleton_tolerance", 0.060)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.45)),
                    max_step=float(ocra_config.get("max_step", 0.060)),
                    finite_difference_step=float(ocra_config.get("finite_difference_step", 1e-4)),
                )
                solver_metrics = dict(functional_metrics or {})
                functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
                functional_metrics.update(solver_metrics)
                orientation_error = float(functional_metrics.get("ocra_orientation_norm_rad", orientation_error))
                elbow_error = float(functional_metrics.get("ocra_skeleton_rmse_m", elbow_error))
            elif orientation_mode == "table_edge_soft_functional" and shoulder_base is not None and elbow_base is not None:
                functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
                anti_config = RETARGET_RUNTIME_CONFIG.get("anti_self_insertion", {})
                q, converged, error, orientation_error, elbow_error, functional_metrics = chain.solve_table_edge_soft_functional_hierarchical(
                    target,
                    desired_rotation,
                    wrist_base,
                    elbow_base,
                    shoulder_base,
                    current_q,
                    robot_shoulder_anchor=robot_shoulder_anchor_base(),
                    elbow_joint_index=robot_elbow_joint_index(),
                    shoulder_joint_index=robot_shoulder_joint_index(),
                    iterations=int(ik_config.get("max_iterations", 120)),
                    primary_damping=float(functional_config.get("primary_damping", 0.045)),
                    secondary_damping=float(functional_config.get("secondary_damping", 0.075)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.014)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.30)),
                    forearm_tolerance=math.radians(float(functional_config.get("forearm_tolerance_deg", 70.0))),
                    upper_arm_tolerance=math.radians(float(functional_config.get("upper_arm_tolerance_deg", 85.0))),
                    anti_self_tolerance=float(anti_config.get("tolerance", 0.02)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.18)),
                    forearm_weight=float(functional_config.get("forearm_weight", 0.20)),
                    upper_arm_weight=float(functional_config.get("upper_arm_weight", 0.10)),
                    included_angle_weight=float(functional_config.get("included_angle_weight", 0.0)),
                    anti_self_weight=float(anti_config.get("weight", 0.70)),
                    joint_regularization_weight=float(functional_config.get("joint_regularization_weight", 0.06)),
                    secondary_step_scale=float(functional_config.get("secondary_step_scale", 0.65)),
                    max_step=float(functional_config.get("max_step", 0.055)),
                    finite_difference_step=float(functional_config.get("finite_difference_step", 1e-4)),
                    secondary_position_guard=float(functional_config.get("secondary_position_guard", 0.045)),
                )
                solver_metrics = dict(functional_metrics or {})
                functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
                transform_for_metrics = chain.fk(q)[0]
                orientation_error = float(
                    np.linalg.norm(orientation_error_vector(transform_for_metrics[:3, :3], desired_rotation))
                )
                functional_metrics["orientation_norm_rad"] = orientation_error
                functional_metrics["anti_self_penalty"] = float(
                    functional_metrics.get("anti_self_penalty", solver_metrics.get("anti_self_penalty", 0.0))
                )
                functional_metrics["anti_self_forearm_alignment"] = float(
                    functional_metrics.get(
                        "anti_self_forearm_alignment",
                        solver_metrics.get("anti_self_forearm_alignment", 1.0),
                    )
                )
                elbow_error = functional_metrics["upper_arm_max_rad"]
            elif orientation_mode == "functional_hierarchical" and shoulder_base is not None and elbow_base is not None:
                functional_config = RETARGET_RUNTIME_CONFIG.get("functional", {})
                q, converged, error, orientation_error, elbow_error, functional_metrics = chain.solve_functional_hierarchical(
                    target,
                    wrist_base,
                    elbow_base,
                    shoulder_base,
                    current_q,
                    robot_shoulder_anchor=robot_shoulder_anchor_base(),
                    elbow_joint_index=robot_elbow_joint_index(),
                    shoulder_joint_index=robot_shoulder_joint_index(),
                    iterations=int(ik_config.get("max_iterations", 120)),
                    primary_damping=float(functional_config.get("primary_damping", 0.045)),
                    secondary_damping=float(functional_config.get("secondary_damping", 0.075)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.014)),
                    forearm_tolerance=math.radians(float(functional_config.get("forearm_tolerance_deg", 25.0))),
                    upper_arm_tolerance=math.radians(float(functional_config.get("upper_arm_tolerance_deg", 35.0))),
                    forearm_weight=float(functional_config.get("forearm_weight", 1.0)),
                    upper_arm_weight=float(functional_config.get("upper_arm_weight", 0.65)),
                    included_angle_weight=float(functional_config.get("included_angle_weight", 0.0)),
                    secondary_step_scale=float(functional_config.get("secondary_step_scale", 0.70)),
                    max_step=float(functional_config.get("max_step", 0.055)),
                    finite_difference_step=float(functional_config.get("finite_difference_step", 1e-4)),
                    secondary_position_guard=float(functional_config.get("secondary_position_guard", 0.035)),
                )
                functional_metrics = compute_functional_metrics(chain, q, wrist_base, elbow_base, shoulder_base)
                orientation_error = functional_metrics["forearm_max_rad"]
                elbow_error = functional_metrics["upper_arm_max_rad"]
            elif orientation_mode == "full_arm_hierarchical" and full_arm_elbow_target is not None:
                full_arm_config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
                q, converged, error, orientation_error, elbow_error = chain.solve_full_arm_hierarchical(
                    target,
                    desired_rotation,
                    full_arm_elbow_target,
                    current_q,
                    elbow_joint_index=int(full_arm_config.get("robot_elbow_joint_index", 3)),
                    iterations=int(ik_config.get("max_iterations", 80)),
                    primary_damping=float(full_arm_config.get("primary_damping", 0.040)),
                    secondary_damping=float(full_arm_config.get("secondary_damping", 0.070)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                    elbow_tolerance=float(full_arm_config.get("elbow_tolerance", 0.055)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
                    elbow_weight=float(full_arm_config.get("elbow_weight", 0.30)),
                    secondary_step_scale=float(full_arm_config.get("secondary_step_scale", 1.0)),
                    max_step=float(full_arm_config.get("max_step", 0.055)),
                )
            elif orientation_mode == "full_arm_aligned" and full_arm_elbow_target is not None:
                full_arm_config = RETARGET_RUNTIME_CONFIG.get("full_arm", {})
                q, converged, error, orientation_error, elbow_error = chain.solve_full_arm(
                    target,
                    desired_rotation,
                    full_arm_elbow_target,
                    current_q,
                    elbow_joint_index=int(full_arm_config.get("robot_elbow_joint_index", 3)),
                    iterations=int(ik_config.get("max_iterations", 80)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                    elbow_tolerance=float(full_arm_config.get("elbow_tolerance", 0.055)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
                    elbow_weight=float(full_arm_config.get("elbow_weight", 0.30)),
                )
            else:
                q, converged, error, orientation_error = chain.solve_pose(
                    target,
                    desired_rotation,
                    current_q,
                    iterations=int(ik_config.get("max_iterations", 80)),
                    position_tolerance=float(ik_config.get("position_tolerance", 0.008)),
                    orientation_tolerance=float(ik_config.get("orientation_tolerance", 0.08)),
                    orientation_weight=float(ik_config.get("orientation_weight", 0.35)),
                )
                elbow_error = None
            orientation_text = "" if orientation_error is None else f" orient={orientation_error:.3f}"
            elbow_text = "" if elbow_error is None else f" elbow={elbow_error:.3f}"
            functional_text = ""
            if functional_metrics is not None:
                forearm_deg = np.round(functional_metrics["forearm_projected_deg"], 1).tolist()
                upper_deg = np.round(functional_metrics["upper_arm_projected_deg"], 1).tolist()
                anti_penalty = float(functional_metrics.get("anti_self_penalty", 0.0))
                anti_alignment = float(functional_metrics.get("anti_self_forearm_alignment", 1.0))
                tool_alignment = float(functional_metrics.get("anti_self_tool_alignment", 1.0))
                ocra_text = ""
                if "ocra_skeleton_rmse_m" in functional_metrics:
                    ocra_text = (
                        f" ocra_rmse={functional_metrics['ocra_skeleton_rmse_m']:.3f}"
                        f" ocra_obj={functional_metrics.get('ocra_objective', 0.0):.3f}"
                    )
                functional_text = (
                    f" forearm_deg={forearm_deg} upper_deg={upper_deg}"
                    f" anti={anti_penalty:.3f} align={anti_alignment:.3f} tool={tool_alignment:.3f}"
                    f"{ocra_text}"
                )
            print(
                f"prepose phase={loaded_trajectory_phase_prefix()}-{observed_phase} "
                f"time={args.prepose_time:.3f}s target={np.round(target, 3)} "
                f"err={error:.4f}{orientation_text}{elbow_text}{functional_text} converged={converged}"
            )
            if args.dry_run:
                return 0 if error < 0.08 else 6
            code = send_position_goal(node, q, args.prepose_duration)
            print(f"trajectory_result={code}")
            return 0 if code in (0, -5) else 5
        positions, diagnostics = build_trajectory(
            chain,
            current_q,
            args.seconds,
            args.fps,
            reconstructed,
            cup_constrained=not args.raw_reconstructed,
        )
        max_error = max(item["error"] for item in diagnostics)
        print(f"points={len(positions)} max_ik_error={max_error:.4f}")
        limited_count = sum(1 for item in diagnostics if item.get("joint_delta_limited"))
        if limited_count:
            print(f"joint_delta_limiter frames={limited_count}")
        functional_items = [item["functional_metrics"] for item in diagnostics if item.get("functional_metrics") is not None]
        if functional_items:
            worst_forearm = max(
                (
                    (item["functional_metrics"]["forearm_max_deg"], item["time"])
                    for item in diagnostics
                    if item.get("functional_metrics") is not None
                ),
                key=lambda value: value[0],
            )
            worst_upper = max(
                (
                    (item["functional_metrics"]["upper_arm_max_deg"], item["time"])
                    for item in diagnostics
                    if item.get("functional_metrics") is not None
                ),
                key=lambda value: value[0],
            )
            max_forearm = worst_forearm[0]
            mean_forearm = sum(item["forearm_max_deg"] for item in functional_items) / len(functional_items)
            max_upper = worst_upper[0]
            mean_upper = sum(item["upper_arm_max_deg"] for item in functional_items) / len(functional_items)
            print(
                "functional_projected_angles "
                f"forearm_mean_deg={mean_forearm:.1f} forearm_max_deg={max_forearm:.1f} "
                f"forearm_worst_time={worst_forearm[1]:.2f}s "
                f"upper_mean_deg={mean_upper:.1f} upper_max_deg={max_upper:.1f} "
                f"upper_worst_time={worst_upper[1]:.2f}s"
            )
            max_anti = max(float(item.get("anti_self_penalty", 0.0)) for item in functional_items)
            min_alignment = min(float(item.get("anti_self_forearm_alignment", 1.0)) for item in functional_items)
            min_tool_alignment = min(float(item.get("anti_self_tool_alignment", 1.0)) for item in functional_items)
            print(
                "anti_self_insertion "
                f"max_penalty={max_anti:.3f} "
                f"min_forearm_alignment={min_alignment:.3f} "
                f"min_tool_alignment={min_tool_alignment:.3f}"
            )
            ocra_items = [item for item in functional_items if "ocra_skeleton_rmse_m" in item]
            if ocra_items:
                mean_ocra_rmse = sum(float(item["ocra_skeleton_rmse_m"]) for item in ocra_items) / len(ocra_items)
                max_ocra_rmse = max(float(item["ocra_skeleton_rmse_m"]) for item in ocra_items)
                mean_ocra_orientation = sum(float(item["ocra_orientation_norm_rad"]) for item in ocra_items) / len(ocra_items)
                max_ocra_orientation = max(float(item["ocra_orientation_norm_rad"]) for item in ocra_items)
                mean_ocra_objective = sum(float(item["ocra_objective"]) for item in ocra_items) / len(ocra_items)
                print(
                    "ocra_baseline "
                    f"skeleton_rmse_mean_m={mean_ocra_rmse:.4f} skeleton_rmse_max_m={max_ocra_rmse:.4f} "
                    f"orientation_mean_rad={mean_ocra_orientation:.3f} orientation_max_rad={max_ocra_orientation:.3f} "
                    f"objective_mean={mean_ocra_objective:.4f}"
                )
        for item in diagnostics[:: max(1, int(args.fps))]:
            orientation_text = ""
            if item["orientation_error"] is not None:
                orientation_text = f" orient={item['orientation_error']:.3f}"
            elbow_text = ""
            if item.get("elbow_error") is not None:
                elbow_text = f" elbow={item['elbow_error']:.3f}"
            functional_text = ""
            if item.get("functional_metrics") is not None:
                functional = item["functional_metrics"]
                forearm_deg = np.round(functional["forearm_projected_deg"], 1).tolist()
                upper_deg = np.round(functional["upper_arm_projected_deg"], 1).tolist()
                anti_penalty = float(functional.get("anti_self_penalty", 0.0))
                anti_alignment = float(functional.get("anti_self_forearm_alignment", 1.0))
                tool_alignment = float(functional.get("anti_self_tool_alignment", 1.0))
                ocra_text = ""
                if "ocra_skeleton_rmse_m" in functional:
                    ocra_text = (
                        f" ocra_rmse={functional['ocra_skeleton_rmse_m']:.3f}"
                        f" ocra_obj={functional.get('ocra_objective', 0.0):.3f}"
                    )
                functional_text = (
                    f" forearm_deg={forearm_deg} upper_deg={upper_deg}"
                    f" anti={anti_penalty:.3f} align={anti_alignment:.3f} tool={tool_alignment:.3f}"
                    f"{ocra_text}"
                )
            print(
                f"{item['time']:>5.2f}s {item['phase']:<19} "
                f"target={np.round(item['target'], 3)} tcp={np.round(item['tcp'], 3)} "
                f"err={item['error']:.4f}{orientation_text}{elbow_text}{functional_text}"
            )
        if args.save_joint_trajectory:
            payload = {
                "schema": "xarm7_joint_trajectory/v1",
                "source_trajectory": args.trajectory_json,
                "retarget_config": args.retarget_config,
                "seconds": float(args.seconds),
                "fps": float(args.fps),
                "joint_names": JOINT_NAMES,
                "positions": [[float(value) for value in row] for row in positions],
                "diagnostics": jsonable(diagnostics),
            }
            out_path = Path(args.save_joint_trajectory)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"joint_trajectory_saved={out_path} points={len(positions)}")
        if args.dry_run:
            return 0 if max_error < 0.08 else 6
        if args.segmented:
            code = send_segmented_trajectory(node, chain, current_q, args.fps)
        elif args.keyframes:
            code = send_keyframe_trajectory(node, positions, args.fps)
        elif moveit2_enabled():
            code = send_moveit_chunked_trajectory(node, positions, args.seconds, args.fps)
        else:
            code = send_trajectory(node, positions, args.seconds, args.fps, wait=not args.no_wait)
        print(f"trajectory_result={code}")
        return 0 if code in (0, -5) else 5
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
