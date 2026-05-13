# D455 HMR Trajectory Contract

The xArm7 retargeting pipeline consumes one stable JSON contract:

```json
{
  "schema": "smplx_d455_reconstructed_right_hand_trajectory/v1",
  "frame": "world",
  "source": "external_rgbd_hmr_or_smplifyx_adapter",
  "seconds": 10.0,
  "fps": 15.0,
  "samples": [
    {
      "time": 0.0,
      "phase": "hmr",
      "right_hand_world": [0.428, -1.432, 1.159],
      "right_hand_camera": [2.149, -0.739, -0.024],
      "pixel": [512.58, 242.88],
      "depth_m": 2.1488,
      "confidence": 1.0
    }
  ]
}
```

Only `samples[*].time`, `samples[*].right_hand_world`, and positive
`samples[*].confidence` are required by `retarget-smplx-hand-to-xarm.py`.
The other fields are diagnostics for debugging D455 projection, depth, and
visibility.

## Simulator Boundary

`docker/reconstruct-smplx-from-d455.py` is the current simulator-side generator.
It projects the observed SMPL-X hand through the Gazebo D455 camera model,
adds small RGB-D measurement noise, and back-projects the point into the world
frame. This proves the data boundary without requiring a full HMR model.

## Real RGB-D HMR / SMPLify-X Boundary

When a real model is available, export one of these formats:

- JSON / NPZ with `joints_world`, `smplx_joints_world`, or `joints3d_world`
  shaped `[frames, joints, 3]`, in meters.
- JSON / NPZ with `joints_camera`, `smplx_joints_camera`, or `joints3d_camera`
  shaped `[frames, joints, 3]`, in the D455 camera frame.
- Optional `joint_names`; include `right_hand`, `right_wrist`, `r_hand`, or
  `r_wrist`. If names are unavailable, pass `--joint-index`.
- Optional `times` or `timestamps`. If missing, pass FPS.

Convert external output:

```bash
./xarm7-gazebo-dev convert-hmr hmr_output.npz smplx_d455_reconstructed_right_hand.json 15
```

Record using the converted trajectory:

```bash
./xarm7-gazebo-dev record-xarm-smplx 16 smplx_d455_reconstructed_right_hand.json
```

The robot control layer should not depend on the HMR model internals. It should
only read this contract.

## RGB-D SMPL-X Fitting Prototype

`xarm7-gazebo-dev fit-smplx-rgbd <recording-dir> [fps]` runs the current
offline RGB-D fitting prototype:

1. Extract paired D455 RGB and raw depth frames from the recording rosbag.
2. Segment the white SMPL-X operator from RGB.
3. Back-project segmented pixels with raw D455 depth into a point cloud.
4. Fit SMPL-X against that point cloud with a Chamfer-style optimizer. The
   default `pose` mode optimizes global alignment plus selected right-arm
   SMPL-X axis-angle pose parameters.
5. Export `smplx_rgbd_pose_fit_right_hand.json` using this same contract.

This is real offline RGB-D pose-level fitting for the right arm. It is still not
a learned HMR model: the pose optimizer uses the known SMPL-X body model,
segmented D455 depth, a local pose prior, and iterative gradient descent.

## Gazebo SMPL-X Animation

`xarm7-gazebo-dev build-smplx-animation 10 5` exports 51 OBJ frames from the
licensed local SMPL-X model. Each frame is a different deformed SMPL-X mesh for
the ready / reach / descend / grasp / lift sequence.

`xarm7-gazebo-dev record-smplx-animation-d455 10 5` loads those frames into
Gazebo and switches the visible frame over time. This makes the D455 observe a
moving SMPL-X mesh, not a single static OBJ. The output recording can then be
processed with:

```bash
./xarm7-gazebo-dev fit-smplx-rgbd recordings/smplx_anim_d455_YYYYMMDD_HHMMSS 2
```

The current Gazebo frame-switching path is intentionally conservative: full
mesh deformation inside Gazebo is approximated with pre-exported OBJ frames.
This is slower than a native skeletal animation plugin, but it keeps the D455
sensor path honest because rendered RGB and depth really come from different
SMPL-X meshes over time.
