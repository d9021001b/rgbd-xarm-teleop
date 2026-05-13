#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash
source /root/ws/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

if ros2 node list 2>/dev/null | grep -qx "/move_group"; then
  echo "move_group=already_running"
  exit 0
fi

python3 - <<'PY'
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
import yaml

config = MoveItConfigsBuilder(
    dof=7,
    robot_type="xarm",
    limited=True,
    ros2_control_plugin="gz_ros2_control/GazeboSimSystem",
    controllers_name="fake_controllers",
).to_moveit_configs().to_dict()
config["use_sim_time"] = True
with open("/tmp/xarm7_move_group_params.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump({"move_group": {"ros__parameters": config}}, handle, sort_keys=False)
PY

nohup ros2 run moveit_ros_move_group move_group \
  --ros-args \
  --params-file /tmp/xarm7_move_group_params.yaml \
  >/tmp/xarm7-move-group.log 2>&1 &
echo "$!" >/tmp/xarm7-move-group.pid

for _ in $(seq 1 20); do
  if ros2 service list 2>/dev/null | grep -qx "/plan_kinematic_path"; then
    echo "move_group=started"
    exit 0
  fi
  sleep 0.5
done

echo "move_group=start_timeout" >&2
tail -n 80 /tmp/xarm7-move-group.log >&2 || true
exit 1
