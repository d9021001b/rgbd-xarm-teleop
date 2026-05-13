#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/xarm_approach}"
RECORD_SECONDS="${2:-12}"

set +u
source /opt/ros/jazzy/setup.bash
source /root/ws/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export DISPLAY="${DISPLAY:-:1}"

mkdir -p "$OUT_DIR"
rm -rf "${OUT_DIR:?}/"*

PRE_POSE=(-1.20 -0.45 0.0 0.85 0.0 0.55 0.0)
APPROACH_POSE=(-2.10 0.55 0.0 1.05 0.0 0.35 0.0)

ros2 control load_controller --set-state active joint_state_broadcaster >/tmp/xarm-load-jsb.log 2>&1 || true
ros2 control load_controller --set-state active xarm7_traj_controller >/tmp/xarm-load-traj.log 2>&1 || true

gz service -s /gui/move_to/pose \
  --reqtype gz.msgs.GUICamera \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'pose: {position: {x: -3.0 y: -2.6 z: 2.0} orientation: {x: 0 y: 0.21 z: 0.26 w: 0.94}}' >/tmp/xarm-camera.log 2>&1 || true

/tmp/send-xarm-goal.py "${PRE_POSE[@]}" 3.0 >/tmp/xarm-prepose.log 2>&1 || true
sleep 1

timeout "$((RECORD_SECONDS + 3))" ros2 bag record \
  -s mcap \
  --storage-preset-profile zstd_fast \
  -o "${OUT_DIR}/d455_rosbag" \
  --topics \
  /tripod_d455/depth/image \
  /tripod_d455/depth/depth_image \
  /tripod_d455/depth/camera_info \
  /tripod_d455/depth/points >/tmp/xarm-approach-rosbag.log 2>&1 &
bag_pid=$!

python3 /tmp/record-d455-10s.py --seconds "$RECORD_SECONDS" --out-dir "$OUT_DIR" >/tmp/xarm-d455-video.log 2>&1 &
d455_pid=$!

ffmpeg -y \
  -video_size 1600x1000 \
  -framerate 15 \
  -f x11grab \
  -i "$DISPLAY" \
  -t "$RECORD_SECONDS" \
  -codec:v libx264 \
  -preset veryfast \
  -pix_fmt yuv420p \
  "${OUT_DIR}/gazebo_xarm_gripper_approach.mp4" >/tmp/xarm-screen-record.log 2>&1 &
ffmpeg_pid=$!

sleep 2
/tmp/send-xarm-goal.py "${APPROACH_POSE[@]}" 7.0 >/tmp/xarm-approach-motion.log 2>&1 || true

wait "$ffmpeg_pid" || true
wait "$d455_pid" || true
wait "$bag_pid" || true

timeout 5 ros2 run tf2_ros tf2_echo link_base link_tcp >"${OUT_DIR}/final_link_tcp_tf.txt" 2>&1 || true
cp /tmp/xarm-prepose.log "${OUT_DIR}/prepose.log" || true
cp /tmp/xarm-approach-motion.log "${OUT_DIR}/motion.log" || true
cp /tmp/xarm-screen-record.log "${OUT_DIR}/screen_record.log" || true
cp /tmp/xarm-approach-rosbag.log "${OUT_DIR}/rosbag.log" || true
cp /tmp/xarm-d455-video.log "${OUT_DIR}/d455_video.log" || true
