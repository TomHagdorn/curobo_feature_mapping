# ur_realsense_mapping

Standalone package: cuRobo volumetric mapping (block-sparse TSDF → ESDF/mesh)
fed by an Intel RealSense on a UR robot. Depends on cuRobo's public Python API
only — the cuRobo checkout stays unmodified and can be updated independently.

## Install

```bash
# from this directory; resolves curobo from ../curobo (see [tool.uv.sources])
uv pip install -e '.[realsense]'
```

ROS 2 mode additionally needs a sourced ROS 2 environment (rclpy, tf2_ros,
message_filters) — not pip-installed.

## Usage

```bash
# Native RealSense .bag, frame-to-model ICP tracking, viser viewer on :8080
ur-rs-map --source bag --bag ~/Documents/recording.bag --visualize

# Live ROS 2 topics (realsense2_camera with align_depth.enable:=true),
# poses from TF (arm-mounted camera: UR driver TF + hand-eye static transform)
ur-rs-map --source ros2 --pose-source tf --world-frame base_link

# Per-frame poses from a file (x y z qw qx qy qz per line)
ur-rs-map --source bag --bag rec.bag --pose-source traj --traj poses.txt
```

Pose sources: `track` (frame-to-model ICP, handheld scans), `static`,
`traj` (file), `tf` (ROS 2 only, for the arm-mounted case).

Outputs: `output_mesh.ply` plus rendered depth/normal/shaded PNGs in the
cuRobo cache dir (or shown in the viser GUI with `--visualize`).

## Layout

- `realsense_bag.py` — .bag frame source (pyrealsense2)
- `ros2_source.py` — ROS 2 topic frame source + TF pose lookup
- `poses.py` — constant-velocity prediction, trajectory file loading
- `cli.py` — mapping loop (`ur-rs-map`)

## cuRobo version notes

Tested against cuRobo main @ `e0b1030` (post warp-1.13 API update; the mapper
feature-integration and mesh-collision fixes are included). One private import
(`curobo._src.perception.mapper.pose_refiner`) is isolated in `cli.py` — check
it after cuRobo updates.
