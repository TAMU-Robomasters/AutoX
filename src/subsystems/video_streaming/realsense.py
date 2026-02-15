"""RealSense video stream subsystem."""

from __future__ import annotations

import sys
from time import perf_counter, time
from typing import List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs
from cfgv import Array

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream
from src.toolbox.globals import config, print

videostream = config.videostream
aiming = config.aiming

MICRO_SECONDS_TO_MILLISECONDS = 1000

align = rs.align(rs.stream.color)


class RealSenseVideoStream(VideoStream):
    """Wrapper around an Intel RealSense camera pipeline.

    Provides color frames, depth queries and simple 3D reprojection helpers.
    """

    def __init__(self) -> None:
        """Initialize the RealSense pipeline and configure streams.

        The constructor will retry until a device connection is successful.
        """
        # frame buffers
        self.color_frame = None
        self.depth_frame = None

        # stream sizes / framerate
        self.color_stream_width = aiming.color_stream_width
        self.color_stream_height = aiming.color_stream_height
        self.depth_stream_width = aiming.depth_stream_width
        self.depth_stream_height = aiming.depth_stream_height
        self.framerate = aiming.stream_framerate

        # depth valid range
        self.depth_min = aiming.min_depth
        self.depth_max = aiming.max_depth

        self.frame_number = 1

        # configure pipeline and start
        self.pipeline = rs.pipeline()
        conf = rs.config()
        conf.enable_stream(
            rs.stream.depth,
            self.depth_stream_width,
            self.depth_stream_height,
            rs.format.z16,
            self.framerate,
        )
        conf.enable_stream(
            rs.stream.color,
            self.color_stream_width,
            self.color_stream_height,
            rs.format.bgr8,
            self.framerate,
        )

        while True:  # loop until successful connection
            try:
                self.cfg = self.pipeline.start(conf)

                sensors = (
                    self.pipeline.get_active_profile().get_device().query_sensors()
                )
                for sensor in sensors:
                    sensor.set_option(rs.option.global_time_enabled, False)
            except Exception as error:
                print("")
                print(error)
                print("trying again")
                continue
            break

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest color frame as a NumPy array.

        Returns:
            numpy.ndarray | None: color image in BGR format, or None on failure.
        """
        try:
            frame = self.pipeline.wait_for_frames()
            # Only works for the D435i IMU frames (if present)
            # runtime.camera.acceleration = frame[2].as_motion_frame().get_motion_data()
            # runtime.camera.gyro = frame[3].as_motion_frame().get_motion_data()

            align_start = perf_counter()
            aligned_frames = align.process(frame)

            self.color_frame = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

            capture_time = frame.get_frame_metadata(
                rs.frame_metadata_value.sensor_timestamp
            )
            frame_time = frame.get_frame_metadata(
                rs.frame_metadata_value.frame_timestamp
            )
            self.capture_time = time() * 1000 - (
                (frame_time - capture_time) / MICRO_SECONDS_TO_MILLISECONDS
            )

            align_end = perf_counter()
            _ = (align_end - align_start) * 1000  # ms, kept for potential debug

            self.frame_number += 1
            if (
                config.hardware.flip_camera
            ):  # rotate 180 degrees and copy to avoid negative strides
                return np.rot90(np.asanyarray(self.color_frame.get_data()), k=2).copy()
            return np.asanyarray(self.color_frame.get_data())
        except Exception as error:
            print(error)
            print("VideoStream: error while getting frames:", error, sys.exc_info()[0])
            print("(retrying)")
            return None

    def get_intrinsics(self) -> Intrinsics:
        """Query and return camera intrinsics.

        Raises:
            RuntimeError: if no device, color sensor, or video profile is available.
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device connected")
        dev = devices[0]

        try:
            color_sensor = dev.query_sensors()[1]
        except Exception:
            raise RuntimeError("Color sensor not found")

        try:
            vsp = color_sensor.get_cam_profiles()[0].as_video_cam_profile()
        except Exception:
            raise RuntimeError("Video cam profile not found")

        intr = vsp.get_intrinsics()

        # camera intrinsics
        dist = np.array(intr.coeffs)
        cam_matrix = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
        )

        return Intrinsics(dist, cam_matrix)

    def get_depth_at(self, u: int, v: int) -> Optional[float]:
        """Return depth in meters at pixel (u, v).

        Returns None when depth is out of configured range.
        """
        if config.hardware.flip_camera:
            u = self.color_stream_width - u
            v = self.color_stream_height - v
        depth: float = self.depth_frame.get_distance(u, v)  # ty:ignore[possibly-missing-attribute]
        if depth < self.depth_min or depth > self.depth_max:
            return None
        return depth

    def get_xyz_at(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """Deproject pixel (u, v) to a 3D point (x, y, z).

        Example:
            x, y, z = video.get_xyz_at(1, 2)
        """
        point_3d = rs.rs2_deproject_pixel_to_point(self.get_intrinsics(), [u, v], depth)
        point_3d = self._retransform_3d_point_to_coordinate_system(point_3d)
        point_3d = self._offset_3d_point_to_camera_center(point_3d)
        return point_3d

    def _retransform_3d_point_to_coordinate_system(
        self, point_3d: Array[float]
    ) -> List[float]:
        """Convert RealSense coordinates to project coordinate system.

        X positive is right, Y positive is forward, Z positive is up.
        """
        point_3d[1], point_3d[2] = point_3d[2], -point_3d[1]
        return point_3d

    def _offset_3d_point_to_camera_center(
        self, point_3d: List[float]
    ) -> Tuple[float, float, float]:
        """Apply small offsets to move point into camera center coordinate system.

        Offsets derived from the RealSense D400 series datasheet (pg. 92).
        """
        point_3d[0] -= 0.0325  # offset color camera X to center of glass
        point_3d[1] += -0.0042  # offset Y to front of glass
        return (point_3d[0], point_3d[1], point_3d[2])

    def __del__(self) -> None:
        """Destructor: stop the RealSense pipeline."""
        print("Closing Realsense Pipeline")
        self.pipeline.stop()
