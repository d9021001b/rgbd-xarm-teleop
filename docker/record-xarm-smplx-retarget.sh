#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/xarm_smplx_retarget}"
RECORD_SECONDS="${2:-12}"
ACTION_SECONDS="${ACTION_SECONDS:-$RECORD_SECONDS}"
RETARGET_CONFIG="${RETARGET_CONFIG:-/tmp/xarm7-smplx-retarget-calibration.json}"
RETARGET_EXECUTION_MODE="${RETARGET_EXECUTION_MODE:-segmented}"
CAPTURE_SECONDS="$RECORD_SECONDS"
if [[ "$ACTION_SECONDS" =~ ^[0-9]+$ && "$CAPTURE_SECONDS" =~ ^[0-9]+$ && "$CAPTURE_SECONDS" -lt $((ACTION_SECONDS + 18)) ]]; then
  CAPTURE_SECONDS="$((ACTION_SECONDS + 18))"
fi

set +u
source /opt/ros/jazzy/setup.bash
source /root/ws/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export DISPLAY="${DISPLAY:-:1}"

mkdir -p "$OUT_DIR"
rm -rf "${OUT_DIR:?}/"*
rm -f /tmp/smplx-d455-reconstructed-right-hand.json

ros2 control load_controller --set-state active joint_state_broadcaster >/tmp/xarm-load-jsb.log 2>&1 || true
ros2 control load_controller --set-state active xarm7_traj_controller >/tmp/xarm-load-traj.log 2>&1 || true

python3 /tmp/attach-cup-to-tcp.py --reset >/tmp/xarm-smplx-retarget-cup-reset.log 2>&1 || true
if [[ "${RETARGET_SAFE_PREPOSE:-true}" == "true" ]]; then
  /tmp/send-xarm-goal.py -1.20 -0.45 0.0 0.85 0.0 0.55 0.0 4.0 \
    >/tmp/xarm-smplx-retarget-safe-prepose.log 2>&1 || true
  sleep 0.3
fi
if [[ -n "${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON:-}" && -f "${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON}" ]]; then
  cp "${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON}" /tmp/smplx-d455-reconstructed-right-hand.json
fi
retarget_config_args=()
if [[ -f "$RETARGET_CONFIG" ]]; then
  retarget_config_args=(--retarget-config "$RETARGET_CONFIG")
fi
if [[ -f /tmp/smplx-d455-reconstructed-right-hand.json && ${#retarget_config_args[@]} -gt 0 ]]; then
  prepose_extra_args=()
  if [[ "$RETARGET_EXECUTION_MODE" == "raw_dense" ]]; then
    prepose_extra_args=(--raw-reconstructed)
  fi
  if [[ "${RETARGET_PREPOSE_ONLY:-false}" == "true" ]]; then
    python3 /tmp/retarget-smplx-hand-to-xarm.py \
      --prepose-only \
      --prepose-time "${RETARGET_PREPOSE_TIME:-0.0}" \
      --prepose-duration "${RETARGET_PREPOSE_DURATION:-3.0}" \
      "${retarget_config_args[@]}" \
      "${prepose_extra_args[@]}" \
      --trajectory-json /tmp/smplx-d455-reconstructed-right-hand.json \
      >/tmp/xarm-smplx-retarget-prepose.log 2>&1 || true
  else
    python3 /tmp/retarget-smplx-hand-to-xarm.py \
      --seconds "${RETARGET_PREPOSE_SECONDS:-1.4}" \
      --fps 10 \
      "${retarget_config_args[@]}" \
      "${prepose_extra_args[@]}" \
      --trajectory-json /tmp/smplx-d455-reconstructed-right-hand.json \
      >/tmp/xarm-smplx-retarget-prepose.log 2>&1 || true
  fi
else
  /tmp/send-xarm-goal.py -1.20 -0.45 0.0 0.85 0.0 0.55 0.0 5.0 >/tmp/xarm-smplx-retarget-prepose.log 2>&1 || true
fi
sleep "${RETARGET_POST_PREPOSE_SETTLE_SECONDS:-0.5}"

GZ_BIN="/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
if [[ -z "${RETARGET_GUI_CAMERA_REQ:-}" ]]; then
  RETARGET_GUI_CAMERA_REQ='pose: {position: {x: -0.55 y: -2.15 z: 1.60} orientation: {x: 0 y: 0.16 z: 0.30 w: 0.94}}'
fi
if [[ -z "${RETARGET_CLOSEUP_CAMERA_REQ:-}" ]]; then
  RETARGET_CLOSEUP_CAMERA_REQ='pose: {name: "retarget_closeup_camera" position: {x: -0.55 y: -2.60 z: 1.55} orientation: {x: 0 y: 0 z: 0.272 w: 0.962}}'
fi
SMPLX_ANIM_MANIFEST="/root/ws/install/xarm_gazebo/share/xarm_gazebo/models/smplx_operator_animation/animation_manifest.json"
if [[ -f "$SMPLX_ANIM_MANIFEST" ]]; then
  python3 /tmp/animate-smplx-frame-sequence.py \
    --manifest "$SMPLX_ANIM_MANIFEST" \
    --hide-static \
    --prepare-only \
    >/tmp/xarm-smplx-retarget-hide-smplx.log 2>&1 || true
  if [[ "${RETARGET_SHOW_SMPLX_ANIMATION:-false}" == "true" && "${RETARGET_SHOW_SMPLX_INITIAL_FRAME:-true}" == "true" ]]; then
    python3 /tmp/animate-smplx-frame-sequence.py \
      --manifest "$SMPLX_ANIM_MANIFEST" \
      --seconds 0.15 \
      --fps "${RETARGET_SMPLX_ANIMATION_FPS:-10}" \
      --time-scale 1.0 \
      --start-delay 0.0 \
      --skip-hide-all \
      --hide-static \
      >>/tmp/xarm-smplx-retarget-hide-smplx.log 2>&1 || true
  fi
else
  "$GZ_BIN" service \
    -s /world/default/set_pose \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 3000 \
    --req 'name: "smplx_operator_visual" position: {x: 0 y: 0 z: -35} orientation: {x: 0 y: 0 z: 0 w: 1}' \
    >/tmp/xarm-smplx-retarget-hide-smplx.log 2>&1 || true
fi

"$GZ_BIN" service \
  -s /gui/move_to/pose \
  --reqtype gz.msgs.GUICamera \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req "$RETARGET_GUI_CAMERA_REQ" \
  >/tmp/xarm-smplx-retarget-camera.log 2>&1 || true
"$GZ_BIN" service \
  -s /world/default/set_pose_vector/blocking \
  --reqtype gz.msgs.Pose_V \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req "$RETARGET_CLOSEUP_CAMERA_REQ" \
  >>/tmp/xarm-smplx-retarget-camera.log 2>&1 || true
sleep 0.5

timeout "$((CAPTURE_SECONDS + 3))" ros2 bag record \
  -s mcap \
  --storage-preset-profile zstd_fast \
  -o "${OUT_DIR}/d455_rosbag" \
  --topics \
  /joint_states \
  /tf \
  /tf_static \
  /tripod_d455/depth/image \
  /tripod_d455/depth/depth_image \
  /tripod_d455/depth/camera_info \
  /tripod_d455/depth/points \
  /retarget_closeup/image >/tmp/xarm-smplx-retarget-rosbag.log 2>&1 &
bag_pid=$!

python3 /tmp/record-d455-10s.py --seconds "$CAPTURE_SECONDS" --out-dir "$OUT_DIR" >/tmp/xarm-smplx-retarget-d455.log 2>&1 &
d455_pid=$!

python3 /tmp/record-retarget-closeup.py \
  --seconds "$CAPTURE_SECONDS" \
  --fps "${RETARGET_CLOSEUP_FPS:-10}" \
  --out-dir "$OUT_DIR" >/tmp/xarm-smplx-retarget-closeup.log 2>&1 &
closeup_pid=$!

ffmpeg -y \
  -video_size 1600x1000 \
  -framerate 15 \
  -f x11grab \
  -i "$DISPLAY" \
  -t "$CAPTURE_SECONDS" \
  -codec:v libx264 \
  -preset veryfast \
  -pix_fmt yuv420p \
  "${OUT_DIR}/gazebo_gui_reference.mp4" >/tmp/xarm-smplx-retarget-screen.log 2>&1 &
ffmpeg_pid=$!

sleep 1
if [[ -n "${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON:-}" && -f "${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON}" ]]; then
  cp "${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON}" /tmp/smplx-d455-reconstructed-right-hand.json
  {
    echo "reconstruction_source=external_rgbd_hmr_or_smplifyx"
    echo "input=${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON}"
    python3 - <<'PY'
import json
with open("/tmp/smplx-d455-reconstructed-right-hand.json", encoding="utf-8") as handle:
    payload = json.load(handle)
print(f"schema={payload.get('schema')}")
print(f"samples={len(payload.get('samples', []))}")
PY
  } >/tmp/xarm-smplx-retarget-reconstruct.log 2>&1
else
  python3 /tmp/reconstruct-smplx-from-d455.py \
    --seconds "$RECORD_SECONDS" \
    --fps 15 \
    --out /tmp/smplx-d455-reconstructed-right-hand.json \
    >/tmp/xarm-smplx-retarget-reconstruct.log 2>&1
fi

{
  echo "cup_attach_disabled=true"
  echo "reason=retargeting_demo_keeps_cup_static"
} >/tmp/xarm-smplx-retarget-cup-attach.log
if [[ -f "$RETARGET_CONFIG" ]]; then
  retarget_config_args=(--retarget-config "$RETARGET_CONFIG")
  cp "$RETARGET_CONFIG" "${OUT_DIR}/retarget_calibration.json" || true
fi
smplx_anim_pid=""
start_smplx_comparison_animation() {
  if [[ "${RETARGET_SHOW_SMPLX_ANIMATION:-false}" != "true" ]]; then
    return 0
  fi
  if [[ ! -f "$SMPLX_ANIM_MANIFEST" ]]; then
    echo "smplx_animation_manifest_missing=$SMPLX_ANIM_MANIFEST" >"${OUT_DIR}/smplx_animation.log"
    return 0
  fi
  local anim_seconds
  anim_seconds="$(python3 - <<PY
action = float("${ACTION_SECONDS}")
scale = float("${RETARGET_SMPLX_ANIMATION_TIME_SCALE:-${RETARGET_TIME_SCALE:-3.0}}")
print(f"{action * scale:.3f}")
PY
)"
  python3 /tmp/animate-smplx-frame-sequence.py \
    --manifest "$SMPLX_ANIM_MANIFEST" \
    --seconds "$anim_seconds" \
    --fps "${RETARGET_SMPLX_ANIMATION_FPS:-10}" \
    --time-scale "${RETARGET_SMPLX_ANIMATION_TIME_SCALE:-${RETARGET_TIME_SCALE:-3.0}}" \
    --start-delay "${RETARGET_SMPLX_ANIMATION_START_DELAY:-0.7}" \
    --skip-hide-all \
    --hide-static \
    >"${OUT_DIR}/smplx_animation.log" 2>&1 &
  smplx_anim_pid=$!
}
if [[ "$RETARGET_EXECUTION_MODE" == "raw_dense" ]]; then
  if [[ -n "${RETARGET_PRECOMPUTED_JOINT_TRAJECTORY:-}" && -f "${RETARGET_PRECOMPUTED_JOINT_TRAJECTORY}" ]]; then
    cp "${RETARGET_PRECOMPUTED_JOINT_TRAJECTORY}" "${OUT_DIR}/precomputed_joint_trajectory.json" || true
    goal_signal="${OUT_DIR}/xarm_goal_accepted.json"
    rm -f "$goal_signal"
    RETARGET_GOAL_ACCEPTED_FILE="$goal_signal" python3 /tmp/retarget-smplx-hand-to-xarm.py \
      --load-joint-trajectory "${RETARGET_PRECOMPUTED_JOINT_TRAJECTORY}" \
      "${retarget_config_args[@]}" \
      >/tmp/xarm-smplx-retarget-motion.log 2>&1 &
    retarget_pid=$!
    for _ in $(seq 1 160); do
      if [[ -f "$goal_signal" ]]; then
        break
      fi
      if ! kill -0 "$retarget_pid" 2>/dev/null; then
        break
      fi
      sleep 0.05
    done
    start_smplx_comparison_animation
    wait "$retarget_pid" || true
  else
    start_smplx_comparison_animation
    python3 /tmp/retarget-smplx-hand-to-xarm.py \
      --seconds "$ACTION_SECONDS" \
      --fps "${RETARGET_FPS:-10}" \
      --raw-reconstructed \
      "${retarget_config_args[@]}" \
      --trajectory-json /tmp/smplx-d455-reconstructed-right-hand.json \
      >/tmp/xarm-smplx-retarget-motion.log 2>&1 || true
  fi
else
  start_smplx_comparison_animation
  python3 /tmp/retarget-smplx-hand-to-xarm.py \
    --seconds "$ACTION_SECONDS" \
    --fps 10 \
    --segmented \
    "${retarget_config_args[@]}" \
    --trajectory-json /tmp/smplx-d455-reconstructed-right-hand.json \
    >/tmp/xarm-smplx-retarget-motion.log 2>&1 || true
fi
if [[ -n "$smplx_anim_pid" ]]; then
  wait "$smplx_anim_pid" || true
fi

wait "$ffmpeg_pid" || true
wait "$closeup_pid" || true
wait "$d455_pid" || true
wait "$bag_pid" || true

timeout 5 ros2 run tf2_ros tf2_echo link_base link_tcp >"${OUT_DIR}/final_link_tcp_tf.txt" 2>&1 || true
cp /tmp/xarm-smplx-retarget-motion.log "${OUT_DIR}/motion.log" || true
cp /tmp/xarm-smplx-retarget-prepose.log "${OUT_DIR}/prepose.log" || true
cp /tmp/xarm-smplx-retarget-safe-prepose.log "${OUT_DIR}/safe_prepose.log" || true
cp /tmp/xarm-smplx-retarget-cup-reset.log "${OUT_DIR}/cup_reset.log" || true
cp /tmp/xarm-smplx-retarget-cup-attach.log "${OUT_DIR}/cup_attach.log" || true
cp /tmp/xarm-smplx-retarget-reconstruct.log "${OUT_DIR}/reconstruct_d455.log" || true
cp /tmp/smplx-d455-reconstructed-right-hand.json "${OUT_DIR}/smplx_d455_reconstructed_right_hand.json" || true
cp /tmp/xarm-smplx-retarget-camera.log "${OUT_DIR}/camera.log" || true
cp /tmp/xarm-smplx-retarget-hide-smplx.log "${OUT_DIR}/hide_smplx.log" || true
if [[ ! -f "${OUT_DIR}/smplx_animation.log" ]]; then
  echo "smplx_animation_disabled=true" >"${OUT_DIR}/smplx_animation.log"
fi
cp /tmp/xarm-smplx-retarget-screen.log "${OUT_DIR}/screen_record.log" || true
cp /tmp/xarm-smplx-retarget-closeup.log "${OUT_DIR}/closeup.log" || true
cp /tmp/xarm-smplx-retarget-rosbag.log "${OUT_DIR}/rosbag.log" || true
cp /tmp/xarm-smplx-retarget-d455.log "${OUT_DIR}/d455_video.log" || true
