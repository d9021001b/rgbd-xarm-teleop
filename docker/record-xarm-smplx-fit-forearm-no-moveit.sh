#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/xarm_smplx_fit_forearm_no_moveit}"
RECORD_SECONDS="${2:-36}"

export RETARGET_CONFIG="${RETARGET_CONFIG:-/tmp/xarm7-smplx-fit-retarget-forearm-no-moveit-config.json}"
export RETARGET_EXECUTION_MODE="${RETARGET_EXECUTION_MODE:-raw_dense}"
export ACTION_SECONDS="${ACTION_SECONDS:-18}"
export RETARGET_SAFE_PREPOSE="${RETARGET_SAFE_PREPOSE:-true}"
export RETARGET_PREPOSE_ONLY="${RETARGET_PREPOSE_ONLY:-true}"
export RETARGET_PREPOSE_TIME="${RETARGET_PREPOSE_TIME:-0.0}"
export RETARGET_PREPOSE_DURATION="${RETARGET_PREPOSE_DURATION:-3.0}"
export EXTERNAL_SMPLX_D455_TRAJECTORY_JSON="${EXTERNAL_SMPLX_D455_TRAJECTORY_JSON:-/tmp/smplx-fit-right-arm-trajectory.json}"
if [[ -z "${RETARGET_GUI_CAMERA_REQ:-}" ]]; then
  export RETARGET_GUI_CAMERA_REQ='pose: {position: {x: 1.35 y: 1.20 z: 1.75} orientation: {x: 0 y: 0.12 z: -0.952 w: 0.290}}'
fi
if [[ -z "${RETARGET_CLOSEUP_CAMERA_REQ:-}" ]]; then
  export RETARGET_CLOSEUP_CAMERA_REQ='pose: {name: "retarget_closeup_camera" position: {x: 1.20 y: 1.05 z: 1.68} orientation: {x: 0 y: 0 z: -0.957 w: 0.292}}'
fi

exec /tmp/record-xarm-smplx-retarget.sh "$OUT_DIR" "$RECORD_SECONDS"
