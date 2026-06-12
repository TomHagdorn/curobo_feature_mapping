# SPDX-License-Identifier: Apache-2.0
"""ROS 2 node: live cuRobo feature mapping with MoveIt 2 export.

Subscribes to synchronized RealSense depth + color + camera_info, poses each
frame from TF (``world_frame`` -> ``camera_frame``), and fuses everything into
cuRobo's block-sparse TSDF. Optionally fuses C-RADIO patch features for
semantic queries.

Interfaces (under the node name, default ``curobo_map_publisher``):

- ``/planning_scene`` (moveit_msgs/PlanningScene, published): periodic diff
  carrying the extracted (decimated) map mesh as CollisionObject
  ``curobo_map`` — MoveIt's move_group applies it as world geometry.
- ``~/publish_map`` (std_srvs/Trigger): extract + publish the mesh now.
- ``~/save_mesh`` (std_srvs/Trigger): save the full-resolution mesh to
  ``mesh_path``.
- ``~/query_features`` (std_srvs/Trigger): match the ``query_prompt``
  parameter against the fused feature map; response message is JSON with
  match count, best score, and centroid. Matched voxels are published on
  ``~/feature_matches`` (sensor_msgs/PointCloud2, fields x y z score) and the
  centroid on ``~/feature_centroid`` (geometry_msgs/PointStamped).
- ``~/feature_query`` (std_msgs/String, subscribed): one-shot query topic —
  same effect as setting ``query_prompt`` and calling ``~/query_features``.

A plain-string query *service* would need a custom .srv interface package
(rosidl codegen via colcon); the Trigger+parameter pair keeps this package
pip-installable. Example session:

    ros2 param set /curobo_map_publisher query_prompt "table"
    ros2 service call /curobo_map_publisher/query_features std_srvs/srv/Trigger
"""

import json

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.types import CameraObservation, DeviceCfg, Pose

from ur_realsense_mapping.ros2_source import Ros2TopicSource

# moveit_msgs only ships with MoveIt (apt: ros-<distro>-moveit). Without it the
# node still maps and answers feature queries; only scene publishing is off.
try:
    from moveit_msgs.msg import PlanningScene

    from ur_realsense_mapping.moveit_export import (
        decimate,
        make_collision_object,
        make_planning_scene_diff,
    )
except ImportError:  # pragma: no cover
    PlanningScene = None


class CuroboMapPublisher(Node):
    def __init__(self):
        super().__init__("curobo_map_publisher")

        p = self.declare_parameter
        p("world_frame", "base_link")
        p("camera_frame", "camera_color_optical_frame")
        p("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        p("color_topic", "/camera/camera/color/image_raw")
        p("info_topic", "/camera/camera/color/camera_info")
        p("device", "cuda:0")
        p("voxel_size", 0.02)
        p("extent", [4.0, 4.0, 2.0])
        p("grid_center", [0.0, 0.0, 0.5])
        p("truncation_distance", 0.0)  # 0 -> 8 * voxel_size
        p("publish_period", 5.0)  # seconds between planning-scene updates
        p("mesh_max_triangles", 15000)
        p("mesh_path", "curobo_map.ply")
        p("enable_features", True)
        p("feature_stride", 5)  # encode every Nth integrated frame
        p("query_prompt", "")
        p("query_top_k", 200)
        p("query_min_score", 0.05)

        self._device = torch.device(self.get_parameter("device").value)
        self._device_cfg = DeviceCfg(device=self._device)
        self._world_frame = self.get_parameter("world_frame").value
        self._camera_frame = self.get_parameter("camera_frame").value

        # Mapper + depth filter are created lazily on the first frame (image
        # size comes from the stream); RADIO loads eagerly so the model
        # download happens at startup, not mid-mapping.
        self._mapper = None
        self._depth_filter = None
        self._intrinsics_t = None
        self._n_integrated = 0
        self._radio = None
        if self.get_parameter("enable_features").value:
            from ur_realsense_mapping.features import RadioFeatures

            self.get_logger().info("Loading C-RADIO (downloads on first use)...")
            self._radio = RadioFeatures(device=str(self._device))
            self.get_logger().info("C-RADIO ready.")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._map_cloud_pub = self.create_publisher(PointCloud2, "~/map_cloud", 1)
        self._scene_pub = None
        if PlanningScene is not None:
            self._scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 1)
        else:
            self.get_logger().warning(
                "moveit_msgs not found (install ros-<distro>-moveit); "
                "planning-scene publishing disabled"
            )
        self._matches_pub = self.create_publisher(PointCloud2, "~/feature_matches", 1)
        self._centroid_pub = self.create_publisher(PointStamped, "~/feature_centroid", 1)

        # Sensor-data QoS: best-effort subscriber matches both reliable and
        # best-effort publishers (camera drivers and bag replays).
        from rclpy.qos import qos_profile_sensor_data as qos

        sync = ApproximateTimeSynchronizer(
            [
                Subscriber(self, Image, self.get_parameter("depth_topic").value, qos_profile=qos),
                Subscriber(self, CameraInfo, self.get_parameter("info_topic").value, qos_profile=qos),
                Subscriber(self, Image, self.get_parameter("color_topic").value, qos_profile=qos),
            ],
            queue_size=30,
            slop=0.05,
        )
        sync.registerCallback(self._on_frames)
        self._sync = sync

        self.create_subscription(String, "~/feature_query", self._on_query_topic, 1)
        self.create_service(Trigger, "~/publish_map", self._srv_publish_map)
        self.create_service(Trigger, "~/save_mesh", self._srv_save_mesh)
        self.create_service(Trigger, "~/query_features", self._srv_query_features)
        self.create_timer(float(self.get_parameter("publish_period").value), self._publish_map)

        self.get_logger().info(
            f"Mapping {self._camera_frame} in {self._world_frame}; "
            f"features={'on' if self._radio else 'off'}"
        )

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #
    def _init_mapper(self, height: int, width: int):
        voxel = float(self.get_parameter("voxel_size").value)
        truncation = float(self.get_parameter("truncation_distance").value) or voxel * 8
        feature_dim = 0
        feature_grid_hw = (0, 0)
        if self._radio is not None:
            probe = self._radio.extract_patch_features(
                torch.zeros((height, width, 3), dtype=torch.uint8, device=self._device)
            )
            feature_grid_hw = (probe.shape[0], probe.shape[1])
            feature_dim = probe.shape[-1]
        cfg = MapperCfg(
            voxel_size=voxel,
            extent_meters_xyz=tuple(self.get_parameter("extent").value),
            grid_center=torch.tensor(
                self.get_parameter("grid_center").value, dtype=torch.float32
            ),
            truncation_distance=truncation,
            depth_maximum_distance=6.0,
            depth_minimum_distance=0.05,
            minimum_tsdf_weight=2.0,
            decay_factor=1.0,
            frustum_decay_factor=1.0,
            enable_static=False,
            num_cameras=1,
            image_height=height,
            image_width=width,
            feature_dim=feature_dim,
            feature_grid_height=feature_grid_hw[0],
            feature_grid_width=feature_grid_hw[1],
            device=str(self._device),
        )
        self._mapper = Mapper(cfg)
        self._depth_filter = FilterDepth(
            image_shape=(height, width),
            depth_minimum_distance=cfg.depth_minimum_distance,
            depth_maximum_distance=cfg.depth_maximum_distance,
            flying_pixel_threshold=0.5,
            bilateral_kernel_size=3,
        )
        self.get_logger().info(
            f"Mapper initialized ({self._mapper.memory_usage_mb():.0f} MB, "
            f"{width}x{height}, feature_dim={feature_dim})"
        )

    def _lookup_pose(self, stamp) -> "Pose | None":
        try:
            tf = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, stamp, timeout=Duration(seconds=0.1)
            )
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return Pose.from_list(
            [t.x, t.y, t.z, q.w, q.x, q.y, q.z], device_cfg=self._device_cfg
        )

    def _on_frames(self, depth_msg, info_msg, color_msg):
        pose = self._lookup_pose(depth_msg.header.stamp)
        if pose is None:
            return  # TF not available yet for this stamp

        if self._mapper is None:
            self._init_mapper(depth_msg.height, depth_msg.width)
            self._intrinsics_t = torch.from_numpy(
                np.asarray(info_msg.k, dtype=np.float32).reshape(3, 3)
            ).to(self._device)

        depth_np = Ros2TopicSource._depth_to_meters(depth_msg)
        rgb_np = Ros2TopicSource._color_to_rgb(color_msg)

        depth_t = torch.nan_to_num(
            torch.from_numpy(depth_np).to(self._device), nan=0.0
        )
        depth_t, _ = self._depth_filter(depth_t.unsqueeze(0))
        rgb_t = torch.from_numpy(rgb_np).to(self._device).unsqueeze(0)

        feature_grid = None
        stride = int(self.get_parameter("feature_stride").value)
        if self._radio is not None and self._n_integrated % max(1, stride) == 0:
            feats = self._radio.extract_patch_features(rgb_t[0])
            feature_grid = feats.to(dtype=torch.float16).contiguous().unsqueeze(0)

        obs = CameraObservation(
            depth_image=depth_t,
            rgb_image=rgb_t,
            intrinsics=self._intrinsics_t.unsqueeze(0),
            pose=Pose(
                position=pose.position.view(1, 3), quaternion=pose.quaternion.view(1, 4)
            ),
            feature_grid=feature_grid,
        )
        self._mapper.integrate(obs)
        self._n_integrated += 1
        if self._n_integrated % 100 == 0:
            self.get_logger().info(f"Integrated {self._n_integrated} frames")

    # ------------------------------------------------------------------ #
    # MoveIt export
    # ------------------------------------------------------------------ #
    def _extract_mesh_np(self):
        mesh = self._mapper.extract_mesh(surface_only=True)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            return None, None
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces)

    def _publish_map_cloud(self):
        """Publish the colored occupied-voxel cloud for RViz."""
        voxels = self._mapper.integrator.extract_occupied_voxels(surface_only=True)
        if len(voxels) == 0:
            return
        centers = voxels.centers.cpu().numpy()
        colors = voxels.colors_uint8().cpu().numpy().astype(np.uint32)
        if len(centers) > 300_000:
            s = len(centers) // 300_000 + 1
            centers, colors = centers[::s], colors[::s]
        # RViz expects 'rgb' as a packed uint32 reinterpreted as float32.
        packed = ((colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]).view(np.float32)
        fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z", "rgb"))
        ]
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self._world_frame)
        data = np.column_stack([centers.astype(np.float32), packed])
        self._map_cloud_pub.publish(point_cloud2.create_cloud(header, fields, data))

    def _publish_map(self) -> str:
        if self._mapper is None or self._n_integrated == 0:
            return "no frames integrated yet"
        self._publish_map_cloud()
        if self._scene_pub is None:
            return "published map_cloud (moveit_msgs missing; no planning scene)"
        vertices, faces = self._extract_mesh_np()
        if vertices is None:
            return "map is empty"
        max_tris = int(self.get_parameter("mesh_max_triangles").value)
        n_full = len(faces)
        vertices, faces = decimate(vertices, faces, max_tris)
        if len(faces) > max_tris:
            self.get_logger().warning(
                f"mesh has {len(faces)} triangles > mesh_max_triangles={max_tris} "
                "(install fast-simplification for decimation); publishing anyway"
            )
        co = make_collision_object(vertices, faces, frame_id=self._world_frame)
        scene = make_planning_scene_diff(co)
        self._scene_pub.publish(scene)
        msg = f"published curobo_map: {len(faces)} triangles (full map: {n_full})"
        self.get_logger().info(msg)
        return msg

    def _srv_publish_map(self, _request, response):
        response.message = self._publish_map()
        response.success = "published" in response.message
        return response

    def _srv_save_mesh(self, _request, response):
        if self._mapper is None or self._n_integrated == 0:
            response.success, response.message = False, "no frames integrated yet"
            return response
        path = self.get_parameter("mesh_path").value
        mesh = self._mapper.extract_mesh(surface_only=False)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            response.success, response.message = False, "map is empty"
            return response
        mesh.save_as_mesh(path)
        response.success = True
        response.message = f"saved {path} ({len(mesh.vertices)} vertices)"
        return response

    # ------------------------------------------------------------------ #
    # Feature queries
    # ------------------------------------------------------------------ #
    def _run_query(self, prompt: str) -> dict:
        if self._radio is None:
            return {"error": "features disabled (enable_features=false)"}
        if self._mapper is None or self._n_integrated == 0:
            return {"error": "no frames integrated yet"}
        if not prompt:
            return {"error": "empty prompt (set the query_prompt parameter)"}

        text_vec = self._radio.encode_text(prompt)[0]
        matched = self._mapper.extract_matching_feature_voxels(
            feature_vector=text_vec,
            top_k=int(self.get_parameter("query_top_k").value),
            surface_only=True,
            minimum_score=float(self.get_parameter("query_min_score").value),
            feature_projector=self._radio.project_features,
        )
        n_voxels = len(matched)
        result = {"prompt": prompt, "blocks": int(matched.block_pool_idx.numel()),
                  "voxels": n_voxels}
        if n_voxels == 0:
            return result

        centers = matched.voxels.centers.cpu().numpy()
        scores = matched.scores_per_voxel(fill_value=0.0).cpu().numpy()
        result["best_score"] = float(matched.block_scores[0])
        centroid = centers.mean(axis=0)
        result["centroid"] = [round(float(c), 4) for c in centroid]

        stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z", "score"))
        ]
        cloud_data = np.column_stack([centers, scores]).astype(np.float32)
        header = Header(stamp=stamp, frame_id=self._world_frame)
        self._matches_pub.publish(point_cloud2.create_cloud(header, fields, cloud_data))

        pt = PointStamped()
        pt.header.stamp = stamp
        pt.header.frame_id = self._world_frame
        pt.point.x, pt.point.y, pt.point.z = (float(c) for c in centroid)
        self._centroid_pub.publish(pt)
        return result

    def _srv_query_features(self, _request, response):
        result = self._run_query(self.get_parameter("query_prompt").value)
        response.success = "error" not in result
        response.message = json.dumps(result)
        return response

    def _on_query_topic(self, msg: String):
        result = self._run_query(msg.data)
        self.get_logger().info(f"feature query: {json.dumps(result)}")


def main():
    rclpy.init()
    node = CuroboMapPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
