#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash
source /root/ws/install/setup.bash
set -u

export DISPLAY="${DISPLAY:-:1}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export GZ_SIM_RESOURCE_PATH="/root/ws/install/xarm_gazebo/share/xarm_gazebo/models:${GZ_SIM_RESOURCE_PATH:-}"

cd /root/ws
exec ros2 launch xarm_gazebo xarm7_beside_table_gazebo.launch.py add_soft_gripper:=true load_controller:=true
