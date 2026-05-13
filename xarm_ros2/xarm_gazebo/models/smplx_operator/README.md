# SMPL-X Operator Mesh

Place a locally generated `smplx_operator.obj` in `meshes/` to use this Gazebo model.
If the OBJ references an `.mtl` file, keep the `.mtl` and UV texture images next
to the source OBJ before installing. Gazebo will load the OBJ and follow the MTL
texture references from the model's `meshes/` directory.

The SMPL-X body model files are not bundled here. Download them from the official
SMPL-X site after accepting its license, then generate the OBJ with:

```bash
python3 docker/export-smplx-operator.py \
  --model-dir /path/to/SMPLX_MODEL_DIR \
  --uv-template /Users/ryanm3/Documents/Codex/2026-05-06/smplx/smplx_uv_2023.zip \
  --out xarm_ros2/xarm_gazebo/models/smplx_operator/meshes/smplx_operator.obj
```

To regenerate the current right-hand-to-cup pose, include the Blender addon's
hand pose library and the built-in preset:

```bash
python3 docker/export-smplx-operator.py \
  --model-dir /Users/ryanm3/Documents/Codex/2026-05-06/smplx/models \
  --uv-template /Users/ryanm3/Documents/Codex/2026-05-06/smplx/smplx_uv_2023.zip \
  --texture /Users/ryanm3/Documents/Codex/2026-05-06/smplx/blender_addon_data/smplx_blender_addon/data/smplx_texture_m_2023.png \
  --hand-poses /Users/ryanm3/Documents/Codex/2026-05-06/smplx/blender_addon_data/smplx_blender_addon/data/smplx_handposes.npz \
  --preset cup-grasp \
  --out xarm_ros2/xarm_gazebo/models/smplx_operator/meshes/smplx_operator.obj
```

For an already exported OBJ with UV material files, install it with:

```bash
xarm7-gazebo-dev install-smplx-operator \
  /Users/ryanm3/Documents/Codex/2026-05-06/smplx/smplx_uv_2023.zip
```

The installer copies `smplx_operator.obj`, adjacent `.mtl` files, and referenced
texture images into `meshes/`, rewriting texture paths to local basenames so
Gazebo can resolve them inside Docker.

For robot simulation, keep using simple collision proxies for physics and the
SMPL-X mesh for visual/HMR alignment. A dense human mesh is too heavy and too
fragile to use directly as a real-time collision body.
