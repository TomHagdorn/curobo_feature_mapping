# ur_realsense_mapping

Standalone package feeding RealSense data into cuRobo's volumetric mapper
(block-sparse TSDF → ESDF/mesh) for a UR robot with an arm-mounted D435i.
Split out of the curobo checkout on 2026-06-12 so cuRobo can update freely;
old in-tree history: curobo repo, branch `thagdorn/archive-mapper`.

## Environment

- Python: `/home/tsp_th/curobo/.venv/bin/python` (has curobo editable from
  `../curobo`, pyrealsense2, and this package editable; CLI: `ur-rs-map`).
- cuRobo: resolved from `../curobo` via `[tool.uv.sources]`; tested at
  upstream main `e0b1030` (post warp-1.13).
- ROS 2 mode needs a sourced ROS environment (rclpy etc., not pip).

## Architecture

- `realsense_bag.py` — native .bag source; yields (depth_m, rgb, K, gyro_samples).
- `ros2_source.py` — live topics + TF pose lookup (arm-mounted path; UNTESTED).
- `poses.py` — gyro quaternion integration, constant-velocity prediction.
- `cli.py` — mapping loop; pose sources: track (ICP, gyro-seeded) / static /
  traj / tf. Contains the one private curobo import
  (`curobo._src.perception.mapper.pose_refiner`) — re-check after curobo updates.

## Key facts learned

- The gyro rotation prior (commit 2f92e98) fixed handheld tracking: D435i bag
  `~/Documents/20260211_150520.bag` went 254 kept/239 lost → 493/493 frames.
  Good params: `--voxel-size 0.015 --truncation-distance 0.12 --max-track-error 0.08`.
- D4xx gyro rates are in a depth-optical-aligned frame; used directly as
  camera body rates.
- Purple blocks in renders = unobserved space, not a bug.
- MoveIt 2 can only consume geometry (mesh CollisionObject / octomap), not
  feature maps. Agreed architecture: ONE cuRobo TSDF (+C-RADIO features),
  geometry exported to MoveIt, semantic queries answered in cuRobo.

## Planned next

- `--features` flag: C-RADIO feature integration (see curobo
  `examples/getting_started/feature_mapping.py`).
- MoveIt 2 publisher node: periodic `mapper.extract_mesh()` →
  `moveit_msgs/CollisionObject`.
- RobotSegmenter (curobo.perception) to mask the arm out of depth before
  integration once camera is arm-mounted.
- Test ros2_source against `ros2 bag play` / live driver.
