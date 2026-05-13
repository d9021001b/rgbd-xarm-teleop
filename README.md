# ROS 2 xArm7 Gazebo on macOS via Ubuntu Container

This keeps macOS out of the ROS/Gazebo dependency stack. Colima provides an
Ubuntu-capable Docker VM, and the container runs ROS 2 Jazzy, MoveIt, Gazebo
Harmonic integration, xArm ROS 2 packages, and a noVNC desktop.

## Commands

```bash
xarm7-gazebo-dev build
xarm7-gazebo-dev start
xarm7-gazebo-dev launch
```

For a detached simulator process:

```bash
xarm7-gazebo-dev launch-bg
xarm7-gazebo-dev sim-logs
```

Open the simulator desktop at:

```text
http://localhost:6080/vnc.html?autoconnect=1&resize=scale
```

The plain `vnc.html` page stops on noVNC's connection screen until you click
`Connect`.

## Repository Contents and External Assets

This repository includes enough assets to launch the Gazebo xArm7 scene and run
the retargeting code path:

- ROS 2 Jazzy / Gazebo Harmonic Docker environment.
- xArm7 ROS 2 packages, launch files, URDF/xacro, MoveIt2 config, and Gazebo
  worlds.
- The table, red cup target, tripod D455-style camera, and static Gazebo
  SMPL-X operator visual.
- The Gazebo SMPL-X visual mesh and texture files under
  `xarm_ros2/xarm_gazebo/models/smplx_operator/meshes/`.
- Retargeting scripts, calibration configs, OCRA-style, SEW-Mimic-style,
  RGB-D DLS, and SEW-functional hybrid baseline configs, and documentation.

The following files are intentionally **not** tracked in git. Users who need
the full HMR / SMPLify-X / mesh-fitting pipeline must download or generate
them locally:

| Asset | Required for | Where to get it | Suggested local path |
|---|---|---|---|
| SMPL-X body model weights, e.g. `SMPLX_NEUTRAL.npz` from `models_smplx_v1_1.zip` | Running true SMPL-X model fitting or regenerating posed SMPL-X meshes | Register and download from the official SMPL-X website after accepting its license: <https://smpl-x.is.tue.mpg.de> | `external/smplx/models/smplx/SMPLX_NEUTRAL.npz` or any path passed as `--model-dir` |
| SMPL-X UV map / texture zip, e.g. `smplx_uv_2023.zip` | Regenerating the Gazebo OBJ/texture assets | Official SMPL-X downloads section | `external/smplx/smplx_uv_2023.zip` |
| VPoser v1.0, e.g. `vposer_v1_0.zip` | SMPLify-X-style pose prior during fitting | Official SMPL-X / SMPLify-X downloads section | `external/smplx/vposer_v1_0/` |
| SMPL-X MANO / FLAME correspondences, e.g. `smplx_mano_flame_correspondences.zip` | Hand/face correspondence utilities and full expressive-body fitting experiments | Official SMPL-X downloads section | `external/smplx/correspondences/` |
| Human video, RGB-D rosbag, or HMR output trajectory JSON | Reproducing a specific human-to-xArm retargeting run | Record your own data or convert an external HMR / SMPLify-X output using `docs/hmr_trajectory_contract.md` | `recordings/<run>/` |
| Generated videos, images, rosbags, and local experiment outputs | Demo artifacts and benchmark plots | Generate locally with the scripts in `docker/` | `recordings/` |
| Research paper PDFs | Literature review and paper-by-paper comparison | Download from each publisher / arXiv / project page. Local notes are kept in `research_papers/.../index.md` and `replicable_methods.md` | `research_papers/.../pdfs/` |

The static Gazebo SMPL-X OBJ included in this repository is a visual asset for
the simulator. It is **not** a substitute for the official SMPL-X parametric
model weights used by SMPL-X / SMPLify-X fitting code.

## Useful Operations

```bash
xarm7-gazebo-dev status
xarm7-gazebo-dev shell
xarm7-gazebo-dev stop
```

View and record the tripod D455 streams:

```bash
xarm7-gazebo-dev view-d455
xarm7-gazebo-dev record-d455 10
```

`view-d455` opens RViz in the noVNC desktop. Use Image displays for
`/tripod_d455/depth/image` and `/tripod_d455/depth/depth_image`, and a
PointCloud2 display for `/tripod_d455/depth/points`.

`record-d455 10` writes a timestamped folder under `recordings/` with
`d455_rgb.mp4`, `d455_depth_colormap.mp4`, `manifest.json`, and a compressed
MCAP ROS bag containing the RGB image, depth image, camera info, and point cloud
topics.

SMPL-X mesh support is scaffolded under
`xarm_ros2/xarm_gazebo/models/smplx_operator/`. A static Gazebo OBJ visual is
tracked in git, so the simulator can show the SMPL-X operator without additional
downloads. If you want to regenerate the OBJ from official SMPL-X model weights,
download the SMPL-X assets separately after accepting the SMPL-X license, then
run:

```bash
python3 docker/export-smplx-operator.py \
  --model-dir external/smplx/models \
  --uv-template external/smplx/smplx_uv_2023.zip \
  --out xarm_ros2/xarm_gazebo/models/smplx_operator/meshes/smplx_operator.obj
```

If you already have an OBJ, MTL, texture set, or a zip such as
`smplx_uv_2023.zip`, install it into the Gazebo model with:

```bash
xarm7-gazebo-dev install-smplx-operator external/smplx/smplx_uv_2023.zip
```

Local SMPL-X support assets are staged under
`xarm_ros2/xarm_gazebo/models/smplx_operator/support/`:
`vposer_v1_0` provides a pose prior for SMPLify-style fitting, while
`smplx_mano_flame_correspondences` provides vertex correspondence tables. These
are auxiliary fitting assets; they do not replace the official SMPL-X body model
weights or an exported pose mesh sequence.

Use the SMPL-X mesh as the visual/HMR alignment body and keep simplified
collision proxies for stable real-time Gazebo physics.

Inside the container, the simulator uses:

```bash
ROS_DOMAIN_ID=42
ros2 launch xarm_gazebo xarm7_beside_table_gazebo.launch.py add_soft_gripper:=true
```

## Scene

- xArm7 is spawned beside the table with a fixed four-finger soft gripper on
  the end effector.
- A red cup target sits on the tabletop for approach / grasp planning tests.
- A SMPL-X operator visual is included beside the xArm7. The old primitive
  mannequin and detached arm proxy are intentionally not spawned.
- An Intel RealSense D455-style RGBD camera is mounted on a tripod on the floor
  in front of the table with a wider FOV and longer depth range for capturing
  the robot, cup, and SMPL-X operator together.
- The tripod camera is bridged to ROS 2 on:
  `/tripod_d455/depth/image`, `/tripod_d455/depth/depth_image`,
  `/tripod_d455/depth/camera_info`, and `/tripod_d455/depth/points`.
- xArm7 retargeting consumes `smplx_d455_reconstructed_right_hand.json`, a
  D455 reconstruction boundary artifact, instead of reading built-in SMPL-X
  keypoints directly. The current simulator reconstructor projects the observed
  SMPL-X hand through the D455 camera model and back-projects the RGB-D
  measurement; replace that stage with RGB-D HMR / SMPLify-X when real human
  video is available.
- The stable contract for that replacement is documented in
  `docs/hmr_trajectory_contract.md`. Use `xarm7-gazebo-dev convert-hmr` to
  convert external HMR / SMPLify-X joints into the retarget JSON contract.
- `xarm7-gazebo-dev build-smplx-animation` exports a 10 second SMPL-X OBJ
  frame sequence for ready/reach/descend/grasp/lift, and
  `xarm7-gazebo-dev record-smplx-animation-d455` plays that sequence inside
  Gazebo so the tripod D455 sees a moving SMPL-X mesh rather than a static OBJ.

## Notes

- The Docker context is switched to `colima-ros2` automatically.
- The existing Dify Colima profile is left alone.
- The image builds from Ubuntu 24.04 arm64 packages for ROS 2 Jazzy.
- The xArm Jazzy branch uses Gazebo Harmonic via `ros_gz_*` and
  `gz_ros2_control`; Gazebo Classic is intentionally not used.
- GUI rendering uses software OpenGL in the container, so Gazebo can be slower
  than native Linux with GPU acceleration.
- The upstream xArm Jazzy package metadata references `sdformat14` and
  `gz-sim8` rosdep keys that are satisfied by ROS vendor packages on this
  image, so the Docker build skips those rosdep keys explicitly.

## Verified Locally

- Colima profile: `ros2`
- Docker context: `colima-ros2`
- Image: `local/ros2-xarm7-gazebo:jazzy`
- noVNC: `http://localhost:6080/vnc.html?autoconnect=1&resize=scale`
- Built packages: `xarm_description`, `xarm_msgs`, `xarm_sdk`, `uf_ros_lib`,
  `xarm_api`, `xarm_controller`, `xarm_gazebo`, `xarm_moveit_config`
- Runtime nodes observed: `/controller_manager`, `/gz_ros_control`,
  `/robot_state_publisher`, `/ros_gz_bridge`
- Runtime topics observed: `/clock`, `/joint_states`, `/robot_description`,
  `/tf`, `/tf_static`, `/tripod_d455/depth/image`,
  `/tripod_d455/depth/depth_image`, `/tripod_d455/depth/camera_info`,
  `/tripod_d455/depth/points`, `/retarget_closeup/image`
- Runtime Gazebo models observed: `red_cup_target`, `smplx_operator_visual`,
  `realsense_d455_tripod`, `UF_ROBOT`
