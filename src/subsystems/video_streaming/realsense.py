"""RealSense video stream subsystem."""

from __future__ import annotations

import sys
from time import perf_counter, time
from typing import List, Optional, Tuple

import numpy as np
from cfgv import Array

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream
from src.toolbox.globals import config, print
from src.types.autoaim import Frame

videostream = config.videostream
aiming = config.aiming

MICRO_SECONDS_TO_MILLISECONDS = 1000

# IMPORTANT: pyrealsense2 is NOT imported at module scope. On Jetson the
# CUDA-enabled librealsense .so initializes a CUDA context at import time, and a
# CUDA context does not survive multiprocessing fork — importing it in the parent
# poisons the forked child (the particle filter's first cudaMalloc then fails
# with "initialization error"). The import is done inside load_threaded_cam(),
# which the engine calls from initialize() in the *child* process, populating the
# module-level `rs` used by the methods below (all called only after the camera
# is loaded). See CLAUDE.md on construct-in-initialize().
rs = None  # set by load_threaded_cam()


class RealSenseVideoStream(VideoStream):
    """Wrapper around an Intel RealSense camera pipeline.

    Provides color frames, depth queries and simple 3D reprojection helpers.
    """

    supports_depth: bool = True

    @property
    def height(self) -> int:
        """Get height."""
        return int(self.color_stream_height)

    @property
    def width(self) -> int:
        """Get width."""
        return int(self.color_stream_width)

    def __init__(self) -> None:
        """Configure stream parameters; defer the pipeline to load_threaded_cam().

        The pipeline is intentionally NOT started here. librealsense initializes
        a CUDA context on ``start()``, and a CUDA context does not survive a
        multiprocessing fork. Engines construct this singleton in the *parent*
        process, so starting here would poison the forked child's CUDA state —
        the particle filter's first ``cudaMalloc`` then fails with
        "initialization error". ``load_threaded_cam()`` (called from the engine's
        ``initialize()``, inside the child process) does the real ``start()``.
        """
        # frame buffers
        self.color_frame = None
        self.depth_frame = None

        # color stream sizes / framerate / exposure (from hardware.camera_*)
        self.color_stream_width = config.hardware.camera_width
        self.color_stream_height = config.hardware.camera_height
        self.framerate = config.hardware.camera_fps
        self.exposure = config.hardware.camera_exposure
        self.gain = config.hardware.camera_gain
        self.gamma = config.hardware.camera_gamma
        self.brightness = config.hardware.camera_brightness

        # depth stream sizes / valid range (from aiming)
        self.depth_stream_width = aiming.depth_stream_width
        self.depth_stream_height = aiming.depth_stream_height
        self.depth_min = aiming.min_depth
        self.depth_max = aiming.max_depth

        self.frame_number = 1

        # Built and started in load_threaded_cam() (child process).
        self.pipeline = None
        self.align = None
        self.rs_intrinsics = None

    def load_threaded_cam(self) -> None:
        """Build and start the RealSense pipeline. Must run in the child process.

        Deferred out of ``__init__`` so that BOTH the ``import pyrealsense2``
        (which initializes a CUDA context on Jetson) and the pipeline ``start()``
        happen in the engine's child process rather than the parent — a CUDA
        context does not survive multiprocessing fork (see CLAUDE.md). Retries
        until a device connection succeeds.
        """
        global rs
        if rs is None:
            import pyrealsense2 as rs  # noqa: F811 — child-process-only import

        self.align = rs.align(rs.stream.color)
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

                device = self.pipeline.get_active_profile().get_device()
                for sensor in device.query_sensors():
                    sensor.set_option(rs.option.global_time_enabled, False)

                # Manual exposure + gain/gamma/brightness from config
                # (auto-exposure disabled). Short exposure cuts rolling-shutter
                # smear; gain/gamma/brightness recover image brightness — gamma
                # lifts the icon's midtones without clipping the bright lightbars.
                color_sensor = device.first_color_sensor()
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, float(self.exposure))
                color_sensor.set_option(rs.option.gain, float(self.gain))
                color_sensor.set_option(rs.option.gamma, float(self.gamma))
                color_sensor.set_option(rs.option.brightness, float(self.brightness))

                # Real color-stream intrinsics, used only for depth deprojection
                # in get_xyz_at(). Captured in-process here; never touches the
                # parent. get_intrinsics() still returns placeholders for PnP.
                self.rs_intrinsics = (
                    self.cfg.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
            except Exception as error:
                print("")
                print(error)
                print("trying again")
                continue
            break

    def get_frame(self) -> Optional[Frame]:
        """Return the latest aligned color frame.

        Returns:
            Frame | None: BGR color image + capture timestamp, or None on failure.
        """
        assert self.pipeline is not None, (
            "Call load_threaded_cam() before get_frame()."
        )
        try:
            timestamp = perf_counter()
            frame = self.pipeline.wait_for_frames()
            # Only works for the D435i IMU frames (if present)
            # runtime.camera.acceleration = frame[2].as_motion_frame().get_motion_data()
            # runtime.camera.gyro = frame[3].as_motion_frame().get_motion_data()

            aligned_frames = self.align.process(frame)

            self.color_frame = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

            # capture_time = frame.get_frame_metadata(
            #     rs.frame_metadata_value.sensor_timestamp
            # )
            # frame_time = frame.get_frame_metadata(
            #     rs.frame_metadata_value.frame_timestamp
            # )
            # self.capture_time = time() * 1000 - (
            #     (frame_time - capture_time) / MICRO_SECONDS_TO_MILLISECONDS
            # )

            self.frame_number += 1
            if (
                config.hardware.flip_camera
            ):  # rotate 180 degrees and copy to avoid negative strides
                data = np.rot90(np.asanyarray(self.color_frame.get_data()), k=2).copy()  # type: ignore[attr-defined]
            else:
                data = np.asanyarray(self.color_frame.get_data())  # type: ignore[attr-defined]
            return Frame(data=data, timestamp=timestamp)
        except Exception as error:
            print(error)
            print("VideoStream: error while getting frames:", error, sys.exc_info()[0])
            print("(retrying)")
            return None

    def get_intrinsics(self) -> Intrinsics:
        """Return placeholder pinhole intrinsics for PnP.

        FAKE VALUES FOR NOW. Querying the real device here is commented out
        because (a) it would touch ``rs`` in the parent process at import time
        (pnp.py reads this at module load) and (b) the rvec we want from PnP is
        the immediate goal; position comes from depth, not PnP. Distortion is
        zero (RealSense color is pre-rectified). Replace with the real query
        below once the rest of the pipeline is validated.
        """
        # ctx = rs.context()
        # devices = ctx.query_devices()
        # if len(devices) == 0:
        #     raise RuntimeError("No RealSense device connected")
        # dev = devices[0]
        # try:
        #     color_sensor = dev.query_sensors()[1]
        # except Exception:
        #     raise RuntimeError("Color sensor not found")
        # vsp = color_sensor.get_cam_profiles()[0].as_video_cam_profile()
        # intr = vsp.get_intrinsics()
        # dist = np.array(intr.coeffs)
        # cam_matrix = np.array(
        #     [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
        # )

        dist = np.zeros(5)  # RealSense color is pre-rectified -> no distortion
        # Non-degenerate placeholder K so cv.solvePnP stays well-posed:
        # rough focal length, principal point at frame center.
        fx = fy = 600.0
        cx = self.color_stream_width / 2.0
        cy = self.color_stream_height / 2.0
        cam_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        return Intrinsics(dist, cam_matrix)

    def get_depth_at(self, u: int, v: int) -> Optional[float]:
        """Return depth in meters at pixel (u, v).

        Returns None when depth is out of configured range.
        """
        if config.hardware.flip_camera:
            u = self.color_stream_width - u
            v = self.color_stream_height - v
        depth: float = self.depth_frame.get_distance(u, v)  # type: ignore[attr-defined]
        if depth < self.depth_min or depth > self.depth_max:
            return None
        return depth

    def get_xyz_at(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """Deproject pixel (u, v) to a 3D point (x, y, z).

        Example:
            x, y, z = video.get_xyz_at(1, 2)
        """
        point_3d = rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, [u, v], depth)
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
        """Destructor: stop the RealSense pipeline if it was started."""
        if getattr(self, "pipeline", None) is not None:
            print("Closing Realsense Pipeline")
            self.pipeline.stop()
