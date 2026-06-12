# SPDX-License-Identifier: Apache-2.0
"""Frame source for live ROS 2 topics (realsense2_camera driver or rosbag2 play).

Subscribes to synchronized depth + color + camera_info and yields
``(depth_m, rgb, intrinsics_3x3, pose_or_None)``. When ``world_frame`` is set,
the camera pose is looked up from TF at each frame's timestamp — this is the
path for the arm-mounted camera, where the UR driver publishes TF and a static
hand-eye calibration links the flange to the camera optical frame.

Requires a sourced ROS 2 environment (rclpy, sensor_msgs, tf2_ros,
message_filters); these are intentionally not pip dependencies.
"""

import numpy as np


class Ros2TopicSource:
    """Iterate frames from ROS 2 image topics.

    Default topics match the realsense2_camera driver with
    ``align_depth.enable:=true``:

    - depth: ``/camera/camera/aligned_depth_to_color/image_raw`` (16UC1, mm)
    - color: ``/camera/camera/color/image_raw`` (rgb8)
    - info:  ``/camera/camera/color/camera_info``
    """

    def __init__(
        self,
        depth_topic: str = "/camera/camera/aligned_depth_to_color/image_raw",
        color_topic: str = "/camera/camera/color/image_raw",
        info_topic: str = "/camera/camera/color/camera_info",
        world_frame: str | None = None,
        camera_frame: str = "camera_color_optical_frame",
        color: bool = True,
        queue_size: int = 30,
        slop_s: float = 0.05,
        timeout_s: float = 10.0,
        first_frame_timeout_s: float = 120.0,
    ):
        import rclpy
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self._rclpy = rclpy
        self._want_color = color
        self._world_frame = world_frame
        self._camera_frame = camera_frame
        self._timeout_s = timeout_s
        self._first_frame_timeout_s = first_frame_timeout_s
        self._queue: list = []

        if not rclpy.ok():
            rclpy.init()
        self._node: Node = rclpy.create_node("ur_realsense_mapping_source")

        self._tf_buffer = None
        if world_frame is not None:
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)

        # Sensor-data QoS (best effort): matches both reliable and best-effort
        # publishers; a reliable subscriber would get nothing from best-effort
        # camera drivers / bag replays.
        qos = qos_profile_sensor_data
        subs = [
            Subscriber(self._node, Image, depth_topic, qos_profile=qos),
            Subscriber(self._node, CameraInfo, info_topic, qos_profile=qos),
        ]
        if color:
            subs.append(Subscriber(self._node, Image, color_topic, qos_profile=qos))
        self._sync = ApproximateTimeSynchronizer(subs, queue_size, slop_s)
        self._sync.registerCallback(self._on_frames)

        self.intrinsics = None
        self.image_height = None
        self.image_width = None

    @staticmethod
    def _depth_to_meters(msg) -> np.ndarray:
        data = np.frombuffer(msg.data, dtype=np.uint16 if msg.encoding == "16UC1" else np.float32)
        depth = data.reshape(msg.height, msg.width).astype(np.float32)
        if msg.encoding == "16UC1":
            depth *= 0.001  # mm -> m
        return depth

    @staticmethod
    def _color_to_rgb(msg) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            img = img[..., ::-1]
        elif msg.encoding not in ("rgb8",):
            raise ValueError(f"unsupported color encoding: {msg.encoding}")
        return np.ascontiguousarray(img[..., :3])

    def _lookup_pose(self, stamp):
        if self._tf_buffer is None:
            return None
        from rclpy.duration import Duration

        try:
            tf = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, stamp, timeout=Duration(seconds=0.2)
            )
        except Exception:
            return None  # TF not available yet for this stamp
        t, q = tf.transform.translation, tf.transform.rotation
        return [t.x, t.y, t.z, q.w, q.x, q.y, q.z]

    def _on_frames(self, depth_msg, info_msg, color_msg=None):
        if self.intrinsics is None:
            k = np.asarray(info_msg.k, dtype=np.float32).reshape(3, 3)
            self.intrinsics = k
            self.image_height = depth_msg.height
            self.image_width = depth_msg.width
        depth_m = self._depth_to_meters(depth_msg)
        rgb = self._color_to_rgb(color_msg) if color_msg is not None else None
        pose = self._lookup_pose(depth_msg.header.stamp)
        self._queue.append((depth_m, rgb, self.intrinsics, pose))

    def __iter__(self):
        import time

        try:
            last_frame_t = time.monotonic()
            got_first = False
            while self._rclpy.ok():
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
                while self._queue:
                    last_frame_t = time.monotonic()
                    got_first = True
                    yield self._queue.pop(0)
                # Generous timeout before the first frame (driver/replay may
                # not be running yet), short one afterwards (bag finished).
                limit = self._timeout_s if got_first else self._first_frame_timeout_s
                if time.monotonic() - last_frame_t > limit:
                    break  # no frames arriving (bag finished / driver stopped)
        finally:
            self._node.destroy_node()
