# SPDX-License-Identifier: Apache-2.0
"""Frame source for native RealSense ``.bag`` recordings."""

from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


class RealsenseBag:
    """Iterate aligned depth (meters) + color frames from a RealSense ``.bag``.

    Yields ``(depth_m, rgb, intrinsics_3x3, gyro_samples)`` where
    ``gyro_samples`` is a list of ``(t_seconds, angular_velocity_xyz)`` tuples
    collected since the previous video frame (empty if the bag has no IMU).
    Depth is converted to meters using the depth scale stored in the bag.

    D4xx gyro data is reported in a frame aligned with the depth optical frame
    (x right, y down, z forward), so the rates can be used directly as
    camera-body rotation rates.
    """

    def __init__(self, bag_path: str, color: bool = True):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required to read .bag files. Install with "
                "`pip install 'ur_realsense_mapping[realsense]'`."
            )
        bag_path = str(Path(bag_path).expanduser())
        if not Path(bag_path).exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

        self._want_color = color
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_path, repeat_playback=False)
        profile = self._pipeline.start(config)

        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)  # process every frame, no real-time drops
        self.duration_s = playback.get_duration().total_seconds()
        self.device_name = (
            device.get_info(rs.camera_info.name)
            if device.supports(rs.camera_info.name)
            else "unknown"
        )

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.has_gyro = any(
            s.stream_type() == rs.stream.gyro for s in profile.get_streams()
        )

        self._align = rs.align(rs.stream.depth) if color else None
        self._intrinsics = None  # filled from the first depth frame
        self.image_height = None
        self.image_width = None

    def _intrinsics_from(self, depth_frame) -> np.ndarray:
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        return np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def __iter__(self):
        pending_gyro = []
        try:
            while True:
                try:
                    frameset = self._pipeline.wait_for_frames(5000)
                except RuntimeError:
                    break  # end of bag

                # Collect gyro samples from the raw frameset (motion data
                # arrives in its own framesets between video frames).
                for f in frameset:
                    if f.is_motion_frame() and f.get_profile().stream_type() == rs.stream.gyro:
                        m = f.as_motion_frame().get_motion_data()
                        pending_gyro.append(
                            (
                                f.get_timestamp() / 1000.0,
                                np.array([m.x, m.y, m.z], dtype=np.float32),
                            )
                        )

                if self._align is not None:
                    frameset = self._align.process(frameset)

                depth_frame = frameset.get_depth_frame()
                if not depth_frame:
                    continue  # IMU-only / incomplete composite

                color_frame = frameset.get_color_frame() if self._want_color else None
                if self._want_color and not color_frame:
                    continue

                if self._intrinsics is None:
                    self._intrinsics = self._intrinsics_from(depth_frame)
                    self.image_height = depth_frame.get_height()
                    self.image_width = depth_frame.get_width()

                depth_m = (
                    np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
                )
                rgb = (
                    np.asanyarray(color_frame.get_data()).astype(np.uint8)
                    if color_frame
                    else None
                )
                yield depth_m, rgb, self._intrinsics, pending_gyro
                pending_gyro = []
        finally:
            self._pipeline.stop()

    @property
    def intrinsics(self) -> np.ndarray:
        if self._intrinsics is None:
            raise RuntimeError("intrinsics unavailable until the first frame is read")
        return self._intrinsics
