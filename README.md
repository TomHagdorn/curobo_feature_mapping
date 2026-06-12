# ur_realsense_mapping

Standalone package: cuRobo volumetric mapping (block-sparse TSDF → ESDF/mesh)
fed by an Intel RealSense on a UR robot. Depends on cuRobo's public Python API
only — the cuRobo checkout stays unmodified and can be updated independently.

## Install

cuRobo is resolved from the sibling checkout `../curobo` (see
`[tool.uv.sources]` in `pyproject.toml`). To update cuRobo, `git pull` /
checkout a commit in `../curobo` — this package never conflicts with it.

```bash
cd ~/workspaces/isaac_ros-dev/src/ur_realsense_mapping

# Recommended: into the existing working venv
uv pip install -e '.[realsense]' --python /home/tsp_th/curobo/.venv/bin/python

# Or into a fresh venv (pulls curobo editable from ../curobo automatically)
uv venv && uv pip install -e '.[realsense]'
```

ROS 2 mode additionally needs a **sourced ROS 2 environment** (rclpy, tf2_ros,
message_filters, sensor_msgs) — these come from ROS, not pip:

```bash
source /opt/ros/humble/setup.bash   # adjust distro
```

## Quick start: map from a .bag recording

```bash
# D435i bag with IMU: gyro rotation prior is used automatically for tracking
ur-rs-map --source bag --bag ~/Documents/20260211_150520.bag \
    --voxel-size 0.015 --truncation-distance 0.12 --max-track-error 0.08 \
    --visualize

# open http://localhost:8080 (VSCode forwards the port / Simple Browser)
```

Useful flags: `--no-gyro` (disable the IMU prior), `--stride N`,
`--depth-only`, `--output mesh.ply`, `--extent X Y Z`, `--grid-center X Y Z`.

## Live ROS 2: arm-mounted camera with TF poses

This is the target setup: per-frame camera poses come from TF
(UR forward kinematics + hand-eye calibration) instead of ICP tracking.

Terminal 1 — UR driver (publishes TF `base_link → tool0`):

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur5e robot_ip:=<ROBOT_IP>
```

Terminal 2 — RealSense driver with depth aligned to color:

```bash
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true pointcloud.enable:=false
```

Terminal 3 — hand-eye calibration as a static TF
(`x y z qx qy qz qw` from your calibration, flange → camera optical frame):

```bash
ros2 run tf2_ros static_transform_publisher \
    --x 0.05 --y 0.0 --z 0.06 --qx 0 --qy 0 --qz 0 --qw 1 \
    --frame-id tool0 --child-frame-id camera_color_optical_frame
```

Terminal 4 — the mapper (same venv, ROS sourced):

```bash
source /opt/ros/humble/setup.bash
ur-rs-map --source ros2 --pose-source tf \
    --world-frame base_link --camera-frame camera_color_optical_frame \
    --visualize
```

## Recording a bag on the moving robot

Use **ros2 bag** (not realsense-viewer's native .bag — that format cannot
contain TF). Record while drivers + the hand-eye static TF are running:

```bash
ros2 bag record -o ur_scan_$(date +%Y%m%d_%H%M%S) \
    /camera/camera/aligned_depth_to_color/image_raw \
    /camera/camera/color/image_raw \
    /camera/camera/color/camera_info \
    /tf /tf_static /joint_states
```

- `/tf` + `/tf_static` carry the full pose chain (UR FK + hand-eye); TF is
  interpolated at each image stamp, so exact rates don't need to match.
- `/joint_states` is a cheap insurance: poses can be recomputed via FK later
  if the TF tree was wrong during recording (e.g. missing hand-eye).
- Depth+color at 30 fps is heavy (~100 MB/s uncompressed); drop the camera to
  15 fps or record `.../compressed` topics if disk becomes the bottleneck.

Replay and map:

```bash
ros2 bag play ur_scan_*/ --clock
ur-rs-map --source ros2 --pose-source tf --world-frame base_link --visualize
```

If TF lookups fail during replay, slow it down (`--rate 0.5`) — the mapper
processes frames as they arrive and drops what it can't pose.

Default topics match the realsense2_camera driver
(`/camera/camera/aligned_depth_to_color/image_raw`,
`/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`);
override with `--depth-topic/--color-topic/--info-topic`.

## Pose sources

| `--pose-source` | poses from | when |
|---|---|---|
| `track` (default) | frame-to-model ICP, gyro-seeded if IMU present | handheld scans |
| `tf` | TF lookup world→camera per frame stamp | arm-mounted (ROS 2 only) |
| `traj` | text file, `x y z qw qx qy qz` per line | offline / precomputed FK |
| `static` | `--initial-pose` for every frame | fixed camera |

## Outputs

- `output_mesh.ply` (`--output`) — for MoveIt 2 CollisionObject export etc.
- ESDF voxel grid (in-process, `mapper.compute_esdf()`) — for cuRobo planning
- rendered depth/normal/shaded PNGs in the cuRobo cache dir, or live viser
  viewer on :8080 with `--visualize`

## Layout

- `realsense_bag.py` — .bag frame source (pyrealsense2), incl. gyro samples
- `ros2_source.py` — ROS 2 topic frame source + TF pose lookup
- `poses.py` — gyro integration, constant-velocity prediction, trajectory files
- `cli.py` — mapping loop (`ur-rs-map`)

## cuRobo version notes

Tested against cuRobo main @ `e0b1030` (post warp-1.13 API update). One
private import (`curobo._src.perception.mapper.pose_refiner`) is isolated in
`cli.py` — check it after cuRobo updates.
