# curobo_feature_mapping

Standalone package: cuRobo volumetric mapping (block-sparse TSDF → ESDF/mesh)
fed by an Intel RealSense on a UR robot. Depends on cuRobo's public Python API
only — the cuRobo checkout stays unmodified and can be updated independently.

## Install

cuRobo is resolved from the sibling checkout `../curobo` (see
`[tool.uv.sources]` in `pyproject.toml`). To update cuRobo, `git pull` /
checkout a commit in `../curobo` — this package never conflicts with it.

For ROS 2 mode the venv Python **must match the ROS distro's Python**
(Jazzy = 3.12), otherwise rclpy's C extensions won't import:

```bash
cd ~/workspaces/isaac_ros-dev/src/curobo_feature_mapping
uv venv --python /usr/bin/python3.12 .venv
uv pip install -e '../curobo[cu13]' -e '.[realsense,moveit]' --python .venv/bin/python
```

Run ROS-facing commands with ROS sourced:

```bash
source /opt/ros/jazzy/setup.bash && source .venv/bin/activate
```

MoveIt itself comes from apt: `sudo apt install ros-jazzy-moveit` (without it
the node still maps and answers feature queries; scene publishing is disabled).

For bag-only usage any Python ≥3.10 venv works, e.g. the existing
`/home/tsp_th/curobo/.venv` (3.11).

## Quick start: live camera, no ROS needed

With a RealSense connected via USB (gyro prior active on IMU models;
stop with Ctrl+C — the mesh still gets saved):

```bash
cfmap --source live \
    --voxel-size 0.015 --truncation-distance 0.12 --max-track-error 0.08 \
    --initial-pose 0 0 0 0.5 -0.5 0.5 -0.5 --grid-center 2 0 0 \
    --visualize
```

Hold the camera level when starting (the initial pose anchors the world
upright), then move slowly and deliberately.

## Quick start: map from a .bag recording

```bash
# D435i bag with IMU: gyro rotation prior is used automatically for tracking
cfmap --source bag --bag ~/Documents/20260211_150520.bag \
    --voxel-size 0.015 --truncation-distance 0.12 --max-track-error 0.08 \
    --visualize

# open http://localhost:8080 (VSCode forwards the port / Simple Browser)
```

Useful flags: `--no-gyro` (disable the IMU prior), `--stride N`,
`--depth-only`, `--output mesh.ply`, `--extent X Y Z`, `--grid-center X Y Z`.

## Live ROS 2: poses from TF

The mapper plugs into whatever setup you already run — it makes no
assumptions about the robot. It only needs, from your system:

1. **Topics**: a depth image aligned to the color intrinsics, the color
   image, and its camera_info (with realsense2_camera that means
   `align_depth.enable:=true`).
2. **TF**: a connected chain from your world frame to the camera optical
   frame at the image timestamps. On an arm-mounted camera that chain is
   robot FK plus a static hand-eye transform — published by your own
   launch setup.

Start your setup as usual, point the parameters at it, and run the mapper
(ROS sourced + the py3.12 venv active):

```bash
cfmap-publisher --ros-args --params-file config/map_publisher.yaml
```

Copy [config/map_publisher.yaml](config/map_publisher.yaml) into your own
config directory and set `world_frame`, `camera_frame`, and the three topic
names to match your system; anything can still be overridden ad hoc with
`-p name:=value`.

The interactive CLI works against the same setup, with flags instead of
parameters:

```bash
cfmap --source ros2 --pose-source tf \
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
cfmap --source ros2 --pose-source tf --world-frame base_link --visualize
```

If TF lookups fail during replay, slow it down (`--rate 0.5`) — the mapper
processes frames as they arrive and drops what it can't pose.

Default topics match the realsense2_camera driver
(`/camera/camera/aligned_depth_to_color/image_raw`,
`/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`);
override with `--depth-topic/--color-topic/--info-topic`.

## MoveIt 2 export + feature queries: `cfmap-publisher`

A ROS 2 node that maps continuously (TF poses, like `--pose-source tf`),
optionally fuses C-RADIO features, and exports to MoveIt 2:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate          # the py3.12 venv (must match the ROS python)

# defaults from a YAML params file, individual overrides on the command line
# (rightmost wins: built-in default < --params-file < -p)
cfmap-publisher --ros-args \
    --params-file config/map_publisher.yaml \
    -p world_frame:=base_link -p camera_frame:=camera_color_optical_frame
```

All settings (frames, topics, voxel size, feature options, ...) are normal
ROS 2 parameters — see [config/map_publisher.yaml](config/map_publisher.yaml)
for the full list; copy it into your own setup's config directory and adapt.
Inspect at runtime with `ros2 param list /curobo_map_publisher`.

- Publishes the decimated map mesh as CollisionObject `curobo_map` in
  `/planning_scene` diffs every `publish_period` seconds (requires MoveIt:
  `sudo apt install ros-jazzy-moveit`). Same-id republish = atomic replace.
- `mesh_max_triangles` (default 15000) keeps FCL fast; full-resolution map
  stays in cuRobo.

Services / query interface (string requests need a custom .srv package, so
queries use a parameter + Trigger):

```bash
# semantic query against the fused C-RADIO features
ros2 param set /curobo_map_publisher query_prompt "table"
ros2 service call /curobo_map_publisher/query_features std_srvs/srv/Trigger
#   -> response JSON: {"prompt": "table", "blocks": .., "voxels": ..,
#                      "best_score": .., "centroid": [x, y, z]}
#   matched voxels -> ~/feature_matches (PointCloud2, x y z score; view in RViz)
#   centroid       -> ~/feature_centroid (PointStamped)

# or one-shot via topic
ros2 topic pub --once /curobo_map_publisher/feature_query std_msgs/msg/String "{data: chair}"

# see it: RViz with fixed frame = world_frame, add two PointCloud2 displays:
#   /curobo_map_publisher/map_cloud        (colored map, republished every publish_period)
#   /curobo_map_publisher/feature_matches  (matched voxels; color by the 'score' field)
rviz2

# force a planning-scene update / save the full mesh
ros2 service call /curobo_map_publisher/publish_map std_srvs/srv/Trigger
ros2 service call /curobo_map_publisher/save_mesh std_srvs/srv/Trigger
```

Feature notes: C-RADIO v3-B downloads via torch.hub on first run
(`pip install -e '.[features]'` for its deps, export `HF_TOKEN` if needed);
`feature_stride` (default 5) controls how often RGB frames are encoded.

## Try it: NVIDIA warehouse demo bag (no robot needed)

End-to-end feature mapping + semantic queries against a public simulated
rosbag — a mobile robot driving a warehouse, with ground-truth `/tf` poses.
This is the best dry run before mounting the camera, since it exercises the
exact `--pose-source tf` path your robot will use.

**1. Download the bag** (~640 MB, needs a free NGC account; the nvblox
quickstart asset bundle). Lands in
`${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_nvblox/quickstart/`:

```bash
NGC_ORG=nvidia NGC_TEAM=isaac
NGC_RESOURCE=isaac_ros_nvblox_assets
NGC_FILENAME=quickstart.tar.gz
VERSION=$(ngc registry resource list "$NGC_ORG/$NGC_TEAM/$NGC_RESOURCE:*" 2>/dev/null \
    | grep -oP '\d+\.\d+\.\d+' | sort -V | tail -1)   # or pick a version from the NGC page
curl -LO "https://api.ngc.nvidia.com/v2/resources/$NGC_ORG/$NGC_TEAM/$NGC_RESOURCE/versions/$VERSION/files/$NGC_FILENAME"
mkdir -p "${ISAAC_ROS_WS}/isaac_ros_assets"
tar -xf "$NGC_FILENAME" -C "${ISAAC_ROS_WS}/isaac_ros_assets" && rm "$NGC_FILENAME"
```

(If you already ran the nvblox quickstart, the bag is there — no download
needed.)

**2. Terminal 1 — the mapping node.** Wait for `C-RADIO ready` then
`Mapping ... features=on`:

```bash
cd ~/workspaces/isaac_ros-dev/src/curobo_feature_mapping
source /opt/ros/jazzy/setup.bash && source .venv/bin/activate
cfmap-publisher --ros-args \
    -p world_frame:=odom \
    -p camera_frame:="front_stereo_camera:left_rgb" \
    -p depth_topic:=/front_stereo_camera/depth/ground_truth \
    -p color_topic:=/front_stereo_camera/left/image_raw \
    -p info_topic:=/front_stereo_camera/depth/camera_info \
    -p extent:="[14.0,14.0,4.0]" -p grid_center:="[0.0,0.0,1.0]" \
    -p voxel_size:=0.04 \
    -p enable_features:=true -p feature_stride:=2 \
    -p publish_period:=10.0
```

**3. Terminal 2 — replay the bag** (once Terminal 1 says `features=on`).
Quarter speed: the sim's float-depth frames are ~3.7 MB and best-effort
transport drops most at full rate.

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play --rate 0.2 \
    ~/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_nvblox/quickstart
```

Terminal 1 should log `Mapper initialized (... feature_dim=768)` then
`Integrated N frames`.

**4. Terminal 3 — RViz.** Set **Fixed Frame** to `odom` (not the default
`map`), then **Add → By topic**:

- `/curobo_map_publisher/map_cloud` → PointCloud2, **Size** `0.04`
- `/curobo_map_publisher/feature_matches` → PointCloud2, **Color Transformer**
  `Intensity`, **Channel Name** `score`, **Size** `0.06`

```bash
source /opt/ros/jazzy/setup.bash && rviz2
```

**5. Terminal 4 — query.** Matched voxels light up in RViz over the map;
optionally export each query as a PLY:

```bash
source /opt/ros/jazzy/setup.bash
ros2 param set /curobo_map_publisher query_save_path "/tmp/match_{prompt}.ply"
ros2 param set /curobo_map_publisher query_prompt "shelving rack"
ros2 service call /curobo_map_publisher/query_features std_srvs/srv/Trigger
```

The first query takes ~10 s (kernel compile), the rest are instant. Try
`"cardboard boxes"`, `"floor"`, `"barrel"`. Scores are SigLIP-space cosines
(~0.12–0.20 is a real match, not 0.9); raise `query_min_score` to drop weak
hits.

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
- `cli.py` — mapping loop (`cfmap`)

## cuRobo version notes

Tested against cuRobo main @ `e0b1030` (post warp-1.13 API update). One
private import (`curobo._src.perception.mapper.pose_refiner`) is isolated in
`cli.py` — check it after cuRobo updates.
