# curobo_feature_mapping

Standalone package feeding RealSense data into cuRobo's volumetric mapper
(block-sparse TSDF → ESDF/mesh) for a UR robot with an arm-mounted D435i.
Split out of the curobo checkout on 2026-06-12 so cuRobo can update freely;
old in-tree history: curobo repo, branch `thagdorn/archive-mapper`.

## Environment

- TWO venvs, by purpose:
  - `/home/tsp_th/curobo/.venv` (py3.11): bag CLI `cfmap`. CANNOT import
    rclpy (ROS Jazzy is py3.12).
  - `<repo>/.venv` (py3.12): ROS node `cfmap-publisher`; use with
    `source /opt/ros/jazzy/setup.bash`. Has curobo + this pkg editable.
- cuRobo: pip name is `nvidia-curobo`, resolved editable from `../curobo`;
  tested at upstream main `e0b1030` (post warp-1.13).
- MoveIt NOT installed as of 2026-06-12 (`sudo apt install ros-jazzy-moveit`
  needed); node degrades gracefully (mapping + queries work, no scene pub).

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

## Feature/MoveIt node (added 2026-06-12, commit c9f1d6a)

`map_publisher.py` / `cfmap-publisher`: live TF-posed integration +
C-RADIO feature fusion (`enable_features`, `feature_stride`) + periodic
PlanningScene diffs. Feature queries: set `query_prompt` param, call
`~/query_features` (std_srvs/Trigger, JSON response; matches on
`~/feature_matches` PointCloud2). String-request services would need a
custom .srv interface pkg (colcon) — deliberately avoided.
Key curobo APIs: `Mapper.extract_matching_feature_voxels(feature_vector,
top_k, minimum_score, feature_projector)` → MatchedVoxels
(voxels.centers, scores_per_voxel()); per-voxel feature export via
`extract_occupied_voxels().features()`. `features.py` is adapted from
curobo's feature_mapping example (RADIO via torch.hub NVlabs/RADIO).

## ROS 2 path verified (2026-06-12, commit d3fc118)

`--source ros2 --pose-source tf` works end-to-end against the nvblox
quickstart sim bag (`~/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_nvblox/quickstart`,
sqlite3, 7.8 s, /tf odom->base_link->front_stereo_camera:left_rgb, depth
32FC1 on /front_stereo_camera/depth/ground_truth): 76 frames -> 318k-vertex
colored mesh. Two bugs found+fixed there: subscriptions MUST use
qos_profile_sensor_data (reliable subs get nothing from best-effort bag
replay), and the first frame needs its own 120 s timeout. Replay big sim
bags at --rate 0.25 — best-effort loopback drops most 3.7 MB depth frames
at full rate. Other local assets: galileo bags have /tf_static only (NVIDIA
poses them with live cuVSLAM); r2b_robotarm (NGC r2b 2024) is the public
arm+TF+joint_states bag, camera external/static.

## Planned next

- RobotSegmenter (curobo.perception) to mask the arm out of depth before
  integration once camera is arm-mounted.
- Test map_publisher (the MoveIt node) against the quickstart bag the same
  way (only the CLI path has seen real frames).
- C-RADIO runtime deps untested (`uv pip install -e '.[features]'`, needs
  torch.hub download + possibly HF_TOKEN).
- Custom srv interface package if string-request query services are wanted.
