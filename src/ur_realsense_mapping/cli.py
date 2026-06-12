# SPDX-License-Identifier: Apache-2.0
"""Build a cuRobo volumetric map from a RealSense camera.

Frames come from either a native RealSense ``.bag`` (``--source bag``) or live
ROS 2 topics (``--source ros2``, also covers ``ros2 bag play``). Each frame is
fused into cuRobo's block-sparse TSDF; at the end an ESDF is computed and a
mesh is exported.

Per-frame camera poses come from one of four sources:

* ``--pose-source track`` (default): KinectFusion-style frame-to-model ICP
  against the TSDF being built. Frame 0 is anchored at ``--initial-pose``; each
  later frame is seeded by a constant-velocity prediction and refined. Suitable
  for handheld scans with smooth motion.
* ``--pose-source static``: every frame uses ``--initial-pose``. Only correct if
  the camera did not move.
* ``--pose-source traj``: per-frame poses (``x y z qw qx qy qz`` lines) from
  ``--traj``.
* ``--pose-source tf`` (ros2 source only): camera pose from TF via
  ``--world-frame``/``--camera-frame``. This is the arm-mounted setup: the UR
  driver publishes TF and a static hand-eye transform links flange to camera.

Usage::

    ur-rs-map --source bag --bag ~/Documents/recording.bag --visualize
    ur-rs-map --source ros2 --pose-source tf --world-frame base_link
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from curobo import runtime

# Not yet exported via curobo.perception; keep this single private import
# isolated here so curobo updates only require touching one line.
from curobo._src.perception.mapper.pose_refiner import (
    BlockSparseRaycastPoseRefiner,
    BlockSparseRaycastRefinerCfg,
)
from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.profiling import CudaEventTimer
from curobo.types import CameraObservation, DeviceCfg, Pose

from ur_realsense_mapping.poses import (
    QUAT_IDENTITY,
    integrate_gyro,
    load_trajectory,
    predict_pose,
    quat_mul,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense volumetric mapping with cuRobo")
    parser.add_argument("--source", choices=["bag", "ros2"], default="bag", help="Frame source")
    parser.add_argument("--bag", type=str, default=None, help="Path to a RealSense .bag recording")
    parser.add_argument(
        "--depth-topic", type=str, default="/camera/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--color-topic", type=str, default="/camera/camera/color/image_raw")
    parser.add_argument("--info-topic", type=str, default="/camera/camera/color/camera_info")
    parser.add_argument(
        "--world-frame", type=str, default=None, help="TF world frame for --pose-source tf"
    )
    parser.add_argument(
        "--camera-frame",
        type=str,
        default="camera_color_optical_frame",
        help="TF camera optical frame for --pose-source tf",
    )
    parser.add_argument("--voxel-size", type=float, default=0.02, help="TSDF voxel size (m)")
    parser.add_argument(
        "--truncation-distance",
        type=float,
        default=None,
        help="TSDF truncation band half-width (m); default 8*voxel_size. Wider "
        "bands give the frame-to-model tracker a larger convergence basin.",
    )
    parser.add_argument(
        "--extent",
        nargs=3,
        type=float,
        default=[8.0, 8.0, 4.0],
        help="World extent in meters (x y z)",
    )
    parser.add_argument(
        "--grid-center",
        nargs=3,
        type=float,
        default=None,
        help="Grid center in world frame (default: biased +Z so an identity-pose "
        "camera at the origin looks into the box)",
    )
    parser.add_argument(
        "--pose-source",
        choices=["track", "static", "traj", "tf"],
        default="track",
        help="Where per-frame camera poses come from",
    )
    parser.add_argument(
        "--initial-pose",
        nargs=7,
        type=float,
        default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        help="Frame-0 / static camera pose as x y z qw qx qy qz",
    )
    parser.add_argument("--traj", type=str, default=None, help="Trajectory file for --pose-source traj")
    parser.add_argument(
        "--no-gyro",
        action="store_true",
        help="Disable the gyro rotation prior for --pose-source track (used "
        "automatically when the bag contains an IMU stream)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth video frame")
    parser.add_argument(
        "--max-track-error",
        type=float,
        default=0.05,
        help="RMS SDF error (m) above which a tracked frame is treated as a "
        "tracking loss (pose held, frame not integrated)",
    )
    parser.add_argument(
        "--max-consecutive-lost",
        type=int,
        default=60,
        help="Stop after this many consecutive frames with tracking loss",
    )
    parser.add_argument("--max-frames", type=int, default=100000, help="Max frames to process")
    parser.add_argument("--depth-only", action="store_true", help="Ignore color, integrate depth only")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output", type=str, default="output_mesh.ply", help="Output mesh path")
    parser.add_argument("--visualize", action="store_true", help="Enable viser visualization")
    return parser.parse_args()


def make_source(args):
    """Return an iterable of (depth_m, rgb, intrinsics, pose_or_None, gyro_samples)."""
    if args.source == "bag":
        if args.bag is None:
            raise ValueError("--source bag requires --bag")
        from ur_realsense_mapping.realsense_bag import RealsenseBag

        bag = RealsenseBag(args.bag, color=not args.depth_only)
        print(
            f"  device: {bag.device_name} | duration: {bag.duration_s:.1f}s "
            f"| depth scale: {bag.depth_scale:.6f} | gyro: {bag.has_gyro}"
        )
        return ((d, c, k, None, g) for d, c, k, g in bag)

    from ur_realsense_mapping.ros2_source import Ros2TopicSource

    if args.pose_source == "tf" and args.world_frame is None:
        raise ValueError("--pose-source tf requires --world-frame")
    src = Ros2TopicSource(
        depth_topic=args.depth_topic,
        color_topic=args.color_topic,
        info_topic=args.info_topic,
        world_frame=args.world_frame if args.pose_source == "tf" else None,
        camera_frame=args.camera_frame,
        color=not args.depth_only,
    )
    return ((d, c, k, p, []) for d, c, k, p in src)


def main():
    args = parse_args()
    if args.pose_source == "tf" and args.source != "ros2":
        raise ValueError("--pose-source tf is only available with --source ros2")
    device = torch.device(args.device)
    device_cfg = DeviceCfg(device=device)

    print(f"Opening frame source: {args.source}")
    frame_iter = iter(make_source(args))

    # Pull the first video frame to learn the image size and intrinsics.
    first = next(frame_iter, None)
    if first is None:
        raise RuntimeError("no video frames received")
    depth0, rgb0, intr0, pose0, gyro0 = first
    H, W = depth0.shape
    print(
        f"  image size: {W}x{H} | fx={intr0[0,0]:.1f} fy={intr0[1,1]:.1f} "
        f"cx={intr0[0,2]:.1f} cy={intr0[1,2]:.1f}"
    )

    grid_center = args.grid_center
    if grid_center is None:
        grid_center = [0.0, 0.0, max(0.0, args.extent[2] / 2.0 - 0.5)]

    truncation_distance = args.truncation_distance or args.voxel_size * 8
    config = MapperCfg(
        voxel_size=args.voxel_size,
        extent_meters_xyz=tuple(args.extent),
        grid_center=torch.tensor(grid_center, dtype=torch.float32),
        truncation_distance=truncation_distance,
        depth_maximum_distance=6.0,
        depth_minimum_distance=0.05,
        minimum_tsdf_weight=2.0,
        decay_factor=1.0,
        frustum_decay_factor=1.0,
        enable_static=False,
        roughness=3.0,
        num_cameras=1,
        image_height=H,
        image_width=W,
        device=args.device,
    )
    mapper = Mapper(config)
    print(f"Mapper initialized: {mapper.memory_usage_mb():.1f} MB | grid_center={grid_center}")

    depth_filter = FilterDepth(
        image_shape=(H, W),
        depth_minimum_distance=config.depth_minimum_distance,
        depth_maximum_distance=config.depth_maximum_distance,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=3,
    )

    refiner = None
    trajectory = None
    if args.pose_source == "track":
        # Use a low minimum TSDF weight so voxels seen only once or twice still
        # count as valid correspondences -- otherwise the model is essentially
        # empty for the refiner during the first few dozen frames (a fresh voxel
        # has weight 1/depth^2 <= 1, well below the mapper's 2.0 surface weight).
        refiner = BlockSparseRaycastPoseRefiner(
            mapper.integrator,
            BlockSparseRaycastRefinerCfg(
                depth_minimum_distance=config.depth_minimum_distance,
                depth_maximum_distance=config.depth_maximum_distance,
                minimum_tsdf_weight=0.1,
            ),
        )
    elif args.pose_source == "traj":
        if args.traj is None:
            raise ValueError("--pose-source traj requires --traj")
        trajectory = load_trajectory(args.traj, device_cfg)

    visualizer = None
    if args.visualize:
        from curobo.viewer import ViserVisualizer

        visualizer = ViserVisualizer(connect_port=8080)
        print("Visualization: http://localhost:8080")

    intrinsics_t = torch.from_numpy(np.asarray(intr0)).to(device)
    initial_pose = Pose.from_list(list(args.initial_pose), device_cfg=device_cfg)

    def filtered_depth(depth_np) -> torch.Tensor:
        depth_t = torch.nan_to_num(torch.from_numpy(depth_np).to(device), nan=0.0)
        depth_t, _ = depth_filter(depth_t.unsqueeze(0))  # -> (1, H, W)
        return depth_t

    def make_obs(depth_t: torch.Tensor, rgb_np, pose: Pose) -> CameraObservation:
        if rgb_np is not None:
            rgb_t = torch.from_numpy(rgb_np).to(device).unsqueeze(0)  # (1, H, W, 3)
        else:
            rgb_t = torch.zeros((1, H, W, 3), dtype=torch.uint8, device=device)
        return CameraObservation(
            depth_image=depth_t,
            rgb_image=rgb_t,
            intrinsics=intrinsics_t.unsqueeze(0),
            pose=Pose(position=pose.position.view(1, 3), quaternion=pose.quaternion.view(1, 4)),
        )

    print(f"\nIntegrating frames (pose source: {args.pose_source})...")
    from tqdm import tqdm

    def pose_is_finite(p: Pose) -> bool:
        return bool(
            torch.isfinite(p.position).all() and torch.isfinite(p.quaternion).all()
        )

    def clone_pose(p: Pose) -> Pose:
        return Pose(
            position=p.position.view(1, 3).clone(),
            quaternion=p.quaternion.view(1, 4).clone(),
        )

    good_pose = None  # last pose accepted by the tracker
    good_pose_prev = None  # the one before that (for constant-velocity prediction)
    had_lock = False  # was the previous frame tracked successfully?
    last_pose = initial_pose  # most recent pose estimate (for viz / final report)
    n_done = 0
    n_track_lost = 0
    consecutive_lost = 0
    integrate_ms = []
    pbar = tqdm()

    # The first video frame was already pulled above; build a stream that yields
    # it first, then the rest of the source.
    # Body-frame rotation accumulated from gyro since the last accepted frame;
    # used to seed ICP through fast rotations (and across lost frames).
    gyro_since_good = QUAT_IDENTITY.copy()
    use_gyro = False

    def all_frames():
        yield depth0, rgb0, intr0, pose0, gyro0
        yield from frame_iter

    for raw_idx, (depth_np, rgb_np, _, tf_pose, gyro_samples) in enumerate(all_frames()):
        # Accumulate gyro before the stride check so strided-out frames still
        # contribute their rotation.
        if args.pose_source == "track" and not args.no_gyro and gyro_samples:
            use_gyro = True
            gyro_since_good = quat_mul(gyro_since_good, integrate_gyro(gyro_samples))
        if raw_idx % args.stride != 0:
            continue
        if n_done >= args.max_frames:
            break

        depth_t = filtered_depth(depth_np)
        last_err = 0.0
        skip_integration = False

        # ---- decide this frame's pose ----
        if args.pose_source == "static":
            pose = initial_pose
        elif args.pose_source == "tf":
            if tf_pose is None:
                continue  # TF not available for this frame yet
            pose = Pose.from_list(tf_pose, device_cfg=device_cfg)
        elif args.pose_source == "traj":
            if n_done >= len(trajectory):
                print("trajectory exhausted; stopping")
                break
            pose = trajectory[n_done]
        else:  # track
            if good_pose is None:
                pose = initial_pose  # anchor frame 0; nothing to track against yet
            else:
                # Seed: gyro rotation prior when available (also valid across
                # lost frames, since rotation keeps being accumulated);
                # otherwise constant-velocity prediction if we had lock last
                # frame, else just the last good pose.
                if use_gyro:
                    dpos = [0.0, 0.0, 0.0]
                    if had_lock and good_pose_prev is not None:
                        dpos = (
                            (good_pose_prev.inverse().multiply(good_pose))
                            .position.view(3)
                            .cpu()
                            .tolist()
                        )
                    seed = good_pose.multiply(
                        Pose.from_list(dpos + gyro_since_good.tolist(), device_cfg=device_cfg)
                    )
                    if not pose_is_finite(seed):
                        seed = good_pose
                elif had_lock and good_pose_prev is not None:
                    seed = predict_pose(good_pose, good_pose_prev)
                    if not pose_is_finite(seed):
                        seed = good_pose
                else:
                    seed = good_pose
                refined, err, _ = refiner.refine_pose(depth_t[0], intrinsics_t, seed)
                last_err = err
                lost = (
                    not np.isfinite(err)
                    or err > args.max_track_error
                    or not pose_is_finite(refined)
                )
                if lost:
                    # Skip integration (a guessed pose would smear the map) but
                    # keep going -- the tracker can re-lock when the camera
                    # returns to a mapped region.
                    n_track_lost += 1
                    consecutive_lost += 1
                    skip_integration = True
                    pose = good_pose
                    had_lock = False
                    if consecutive_lost >= args.max_consecutive_lost:
                        print(
                            f"\nLost tracking for {consecutive_lost} consecutive "
                            f"frames; stopping integration."
                        )
                        break
                else:
                    consecutive_lost = 0
                    pose = refined

        # ---- integrate ----
        if not skip_integration:
            obs = make_obs(depth_t, rgb_np, pose)
            timer = CudaEventTimer().start()
            mapper.integrate(obs)
            dt = timer.stop()
            integrate_ms.append(dt * 1000.0)
            if len(integrate_ms) > 20:
                integrate_ms = integrate_ms[-20:]
            n_done += 1
            if args.pose_source == "track":
                good_pose_prev = good_pose
                good_pose = clone_pose(pose)
                had_lock = True
                gyro_since_good = QUAT_IDENTITY.copy()

        last_pose = pose

        pbar.update(1)
        pbar.set_postfix(
            kept=n_done,
            integrate_ms=f"{np.mean(integrate_ms):.1f}" if integrate_ms else "-",
            track_err_mm=f"{last_err * 1000:.1f}",
            track_lost=n_track_lost,
        )

        if visualizer and not skip_integration and n_done % 20 == 0:
            voxels = mapper.integrator.extract_occupied_voxels(surface_only=False)
            if len(voxels) > 0:
                centers = voxels.centers
                colors = voxels.colors_uint8()
                if len(centers) > 100_000:
                    s = max(1, len(centers) // 100_000)
                    centers, colors = centers[::s], colors[::s]
                visualizer.add_point_cloud(
                    pointcloud=centers.cpu().numpy(),
                    colors=colors.cpu().numpy(),
                    point_size=args.voxel_size,
                    name="/reconstruction",
                )
            del voxels
            torch.cuda.empty_cache()
            visualizer.add_frame("/cameras/current", obs.pose, scale=0.1)

    pbar.close()
    print(
        f"\nIntegrated {n_done} frames"
        + (f" ({n_track_lost} with tracking loss)" if n_track_lost else "")
    )
    if args.pose_source == "track":
        p = last_pose.position.view(3).cpu().numpy()
        print(f"Final tracked camera position: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] m")
    if n_done == 0:
        print("Nothing integrated; aborting.")
        return

    # Render from the first camera pose (first call compiles a kernel; slow).
    print("\nRendering from the first camera pose (first call compiles a kernel)...")
    depth_colormap = mapper.render_depth_colormap(intrinsics_t, initial_pose, (H, W))
    normal_colormap = mapper.render_normal_colormap(intrinsics_t, initial_pose, (H, W))
    shaded = mapper.render_shaded(intrinsics_t, initial_pose, (H, W))

    if visualizer:
        server = visualizer._server
        with server.gui.add_folder("Rendered Views"):
            server.gui.add_image(depth_colormap.cpu().numpy(), label="Depth")
            server.gui.add_image(normal_colormap.cpu().numpy(), label="Normals")
            server.gui.add_image(shaded.cpu().numpy(), label="Shaded")
    else:
        import imageio.v3 as iio

        out = Path(runtime.cache_dir) / "examples" / "realsense_mapping"
        out.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(out / "rendered_depth.png"), depth_colormap.cpu().numpy())
        iio.imwrite(str(out / "rendered_normals.png"), normal_colormap.cpu().numpy())
        iio.imwrite(str(out / "rendered_shaded.png"), shaded.cpu().numpy())
        print(f"Saved renders to: {out}")

    print("\nComputing ESDF...")
    voxel_grid = mapper.compute_esdf()
    if voxel_grid.feature_tensor is not None:
        print(
            f"  ESDF shape: {tuple(voxel_grid.feature_tensor.shape)}, "
            f"voxel_size: {voxel_grid.voxel_size:.4f}m"
        )

    print("\nExtracting mesh...")
    mesh = mapper.extract_mesh(surface_only=False)
    if mesh.vertices is not None and len(mesh.vertices) > 0:
        mesh.save_as_mesh(args.output)
        print(f"Saved mesh: {args.output} ({len(mesh.vertices):,} vertices)")
    else:
        print("No mesh extracted (empty reconstruction)")

    if visualizer:
        voxels = mapper.integrator.extract_occupied_voxels(surface_only=False)
        if len(voxels) > 0:
            centers = voxels.centers
            colors = voxels.colors_uint8()
            if len(centers) > 100_000:
                s = max(1, len(centers) // 100_000)
                centers, colors = centers[::s], colors[::s]
            visualizer.add_point_cloud(
                pointcloud=centers.cpu().numpy(),
                colors=colors.cpu().numpy(),
                point_size=args.voxel_size,
                name="/reconstruction",
            )
        del voxels
        torch.cuda.empty_cache()
        print("\nVisualization running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
