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

## Motion planning (added 2026-06-18)

Planning runs in the SAME node/process as the mapper (shares the live in-GPU
map — mesh re-extracted before each plan, no serialization), but is split
across files so `map_publisher.py` stays mapping+queries only (~520 lines):

- `motion_planner.py` = `ArmPlanner`: wraps cuRobo's planner (the only cuRobo
  planning import).
- `planning_node.py` = `PlanningServer` + `create_planning_server(...)`: owns
  the ROS glue (the action/service, `/joint_states` cache, controller client,
  result->JointTrajectory). The node composes it via injected callables
  (`mesh_provider=self._extract_mesh_np`, `query_fn=self._run_query`), so the
  map source / prompt resolver are decoupled (a second node could pass others).

Gated by `enable_planning` (default True); `create_planning_server` returns None
(node keeps mapping/querying) if the interfaces/control_msgs/robot config are
missing. `main()` spins a `MultiThreadedExecutor` (action server + client need it).

- `~/move_to_pose` (curobo_feature_mapping_interfaces/MoveToPose, **action**):
  goal is EITHER a `target_pose` (PoseStamped; TF'd into `world_frame`) OR a
  `prompt` (matched in the feature map via `_run_query`, aim a downward
  `plan_standoff` pose above the centroid). `execute:=true` runs it on the
  controller. Feedback: phase / progress / tracking_error.
- `~/plan_to_pose` (curobo_feature_mapping_interfaces/PlanToPose, **service**):
  plan-only, returns the JointTrajectory. Verified end-to-end over ROS (no sim):
  61-pt trajectory, ~37 ms plan.
- Execution forwards to `control_msgs/action/FollowJointTrajectory` on
  `controller_action` (sim: `/joint_trajectory_controller/follow_joint_trajectory`).
- `move_to_pose` `grasp:=true` plans a grasp MOTION (approach->grasp->lift) via
  cuRobo `MotionPlanner.plan_grasp` instead of a single reach
  (`ArmPlanner.plan_grasp_to_pose` concatenates the 3 interpolated segments).
  Verified plan-only over ROS: 143-pt traj, ~0.12 s.

**cuRobo has NO grasp pose detection** — `plan_grasp` plans the motion given a
grasp pose you supply; it does not analyze geometry to propose grasps. For
"grab the red box" we synthesize a top-down grasp from the feature query: x,y
from the matched centroid, z from the matched **top surface** (`top_z`, added to
the query JSON) + `grasp_clearance` (the object is in the collision map and
there is no gripper to reach inside it, so the contact pose sits just above the
surface). Aiming at the centroid instead lands the tool mid-volume -> in
collision -> "Goalset planning returned None". For real grasp detection bolt on
Contact-GraspNet / AnyGrasp and feed candidates as a `GoalToolPose` goalset.
The **sim UR5e has no gripper** (only the wrist camera), so nothing is actually
picked up — the tool descends onto the object and lifts away.

Goal-height rules live in `planning_node._plan` (NOT `_resolve_goal`, which now
returns the raw target + an `is_prompt` flag): grasp -> +`grasp_clearance`;
reach to a prompt -> +`plan_standoff`; reach to an explicit pose -> as given.
(Earlier bug: grasp prompts double-added `plan_standoff` because the standoff
fallback fired even when grasp passed standoff=0.)

Verified 2026-06-18 end-to-end in a synthetic-RGBD harness (gz Gazebo would not
run on the dev host: gz-transport multicast discovery is broken and the fix
`sudo ip link set lo multicast on` needs root). A fake downward camera over a
table+red-cube fed the real node: mapping integrated -> 2088-tri mesh world ->
`plan_to_pose` reach (61 pts, ~0.07 s) and `move_to_pose` grasp (143 pts,
~0.38 s) both succeed; execution streams feedback to a stub
FollowJointTrajectory server; the `~/query` "the red box" centroid lands on the
cube and the full prompt->grasp->execute path returns success. Still unrun
against actual Gazebo: real controller tracking + the gz camera path.

Scan->stop->prompt-grab demo: `grasp_demo.launch.py` (sim +
`enable_planning:=true enable_features:=true` + auto-scan), then
`ros2 action send_goal .../move_to_pose ... "{prompt: 'the red box', grasp: true, execute: true}"`.

`motion_planner.py` = `ArmPlanner`, the only place importing cuRobo's planner
(`curobo.motion_planner.MotionPlanner/MotionPlannerCfg`, `curobo.scene.Scene/Mesh`,
`GoalToolPose.from_poses`, `JointState.from_position`, result
`get_interpolated_plan()`; interpolated tensors are (1,1,T,dof) — squeeze).
Re-check after curobo updates.

Key gotcha: **cuRobo ships only ur10e**, sim defaults to **ur5e**. Generated
`config/ur5e.yml` once with `scripts/gen_robot_cfg.py` (RobotBuilder fits
collision spheres from the ur_description meshes). Reproduce (persist the URDF
in config/ -- its abs path is baked into ur5e.yml as `urdf_path`):
`xacro $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro
ur_type:=ur5e name:=ur > config/ur5e.urdf` then
`python scripts/gen_robot_cfg.py --urdf $(pwd)/config/ur5e.urdf --mesh-root
$(ros2 pkg prefix ur_description)/share --out config/ur5e.yml`. `asset_root_path`
in the yml points at `/opt/ros/jazzy/share` (ur_description meshes). The generated
config has `base_link: world` (xacro adds a `world` root) but `world->base_link`
is identity, so planning in `world` == planning in `base_link` (the map frame).
`robot_config` param (abs path; "" -> bundled ur5e.yml). `MotionPlannerCfg.create`
joins a *relative* robot arg onto cuRobo's config dir — pass an ABSOLUTE path.

NOT yet exercised against the running Gazebo sim: plan **execution** on the
controller and the **prompt** path (need the full sim + a scanned feature map).

## Planned next

- RobotSegmenter (curobo.perception) to mask the arm out of depth before
  integration once camera is arm-mounted.
- Test map_publisher (the MoveIt node) against the quickstart bag the same
  way (only the CLI path has seen real frames).
- C-RADIO runtime deps untested (`uv pip install -e '.[features]'`, needs
  torch.hub download + possibly HF_TOKEN).
- Exercise `~/move_to_pose` execution + prompt path against the running Gazebo
  sim (`demo.launch.py enable_planning:=true`, scan, then send a goal).
- Per-frame robot self-collision in the map: the wrist camera also images the
  arm; RobotSegmenter (already on the list) would stop the arm being mapped as
  an obstacle the planner then avoids.
