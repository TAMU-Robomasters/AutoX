"""USB camera video stream subsystem."""

from __future__ import annotations

import queue
import threading

import cv2 as cv
import mrcal
import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream

# project imports
from src.toolbox.globals import path_to, print, config

import time

from src.types.autoaim import Frame


class BufferlesCvCapture:
    """Threaded OpenCV capture that keeps only the most recent frame.

    Continuously reads frames from the provided GStreamer pipeline in a
    background thread and stores only the newest frame in a queue so callers
    always receive the most recent image.
    """

    def __init__(self, index: int) -> None:
        """Create and start the reader thread.

        Args:
            index: Index of the USB camera to open.
        """
        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, config.hardware.camera_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.hardware.camera_height)
        # V4L2: 1 = manual, 3 = auto. Must go before CAP_PROP_EXPOSURE or the
        # exposure write is silently ignored while auto-exposure is on.
        self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv.CAP_PROP_EXPOSURE, config.hardware.camera_exposure)
        self.cap.set(cv.CAP_PROP_FPS, config.hardware.camera_fps)

        if not self.cap.isOpened():
            raise Exception("Could not open video.")
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self) -> None:
        """Continuously read frames and keep the latest one in the queue."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame.")
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self) -> np.ndarray:
        """Return the most recent frame from the queue."""
        return self.q.get()


class USBCamVideoStream(VideoStream):
    """USB camera video stream using a GStreamer pipeline and OpenCV.

    Exposes `get_frame()` and `get_intrinsics()` to match the
    `VideoStream` interface.
    """

    def __init__(self, index: int) -> None:
        """Load pinhole calibration; defer camera open to load_threaded_cam().

        Two lens-correction modes (config.hardware.lens_correction_mode):
          "warp"      — load the precomputed mapxy and remap every frame onto the
                        pinhole model in get_frame().
          "unproject" — leave frames raw; load the splined cameramodel so detected
                        pixel points can be undistorted on demand (see
                        distorted_to_pinhole). Cheaper: only PnP corners are
                        transformed, not the whole image.
        Both modes share the same pinhole K (camera_matrix.pkl) + zero distortion.
        """
        # camera_intrinsics_path is the camera's intrinsics dir; the generate
        # script drops every artifact (camera_matrix.pkl, dist.pkl, mapxy.npy,
        # splined.cameramodel, ...) directly in it — flat, no subdir.
        intrinsics_dir = (
            f"{path_to.calibration_presets}/{config.hardware.camera_intrinsics_path}"
        )
        self.intrinsics: Intrinsics = Intrinsics(
            np.load(f"{intrinsics_dir}/dist.pkl", allow_pickle=True),
            np.load(f"{intrinsics_dir}/camera_matrix.pkl", allow_pickle=True),
        )
        self._mode = config.hardware.lens_correction_mode

        if self._mode == "unproject":
            # Detect on the raw frame; transform only the points fed to PnP.
            self._splined = mrcal.cameramodel(f"{intrinsics_dir}/splined.cameramodel")
            self._splined_lensmodel, self._splined_intrinsics = (
                self._splined.intrinsics()
            )
            k = self.intrinsics.camera_matrix
            # mrcal pinhole intrinsics_data layout: [fx, fy, cx, cy].
            self._pinhole_intrinsics = np.array(
                [k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64
            )
            # Raw frames keep the native camera resolution.
            self._width = int(config.hardware.camera_width)
            self._height = int(config.hardware.camera_height)
        else:
            # Per-pixel map that reprojects raw frames onto the pinhole model.
            # Built by utils/camera_calibration/generate_pinhole_from_mrcal.py.
            self._mapxy: np.ndarray = np.load(f"{intrinsics_dir}/mapxy.npy")
            h, w = self._mapxy.shape[:2]
            self._height = int(h)
            self._width = int(w)

        self.index = index
        self.cap = None

    def load_threaded_cam(self):
        try:
            self.cap = BufferlesCvCapture(self.index)
        except Exception as e:
            print(f"Error in BufferlesCvCapture: {e}")
            raise e

    def get_frame(self):
        """Return the most recent frame.

        In "warp" mode the raw frame is remapped onto the pinhole model. In
        "unproject" mode the raw (distorted) frame is returned untouched —
        detection runs on it and only the PnP points are later undistorted.
        """
        assert self.cap, ("Please run load_threaded_cam before getting frame")

        timestamp = time.perf_counter()
        raw = self.cap.read()
        if self._mode == "unproject":
            frame = raw
        else:
            frame = mrcal.transform_image(raw, self._mapxy)

        return Frame(data=frame, timestamp=timestamp)

    def distorted_to_pinhole(self, points: np.ndarray) -> np.ndarray:
        """Map raw-image pixel observations to equivalent pinhole pixels.

        Unprojects the points through the splined lens model and reprojects
        them through the pinhole model, so OpenCV PnP can treat the result as a
        distortion-free pinhole observation. Only valid in "unproject" mode.

        Args:
            points: (N, 2) array of pixel coordinates in the raw frame.

        Returns:
            (N, 2) array of pixel coordinates in the pinhole image.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rays = mrcal.unproject(pts, self._splined_lensmodel, self._splined_intrinsics)
        return mrcal.project(rays, "LENSMODEL_PINHOLE", self._pinhole_intrinsics)

    def camera_points_to_raw_pixels(self, points_cam: np.ndarray) -> np.ndarray:
        """Project camera-frame 3D points onto the raw distorted image.

        Used to draw PnP-reprojected overlays (e.g. the outer panel box) on the
        raw frame shown in "unproject" mode, so they line up with the
        distorted image. Only valid in "unproject" mode.

        Args:
            points_cam: (N, 3) points in the OpenCV camera frame.

        Returns:
            (N, 2) pixel coordinates in the raw frame.
        """
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        return mrcal.project(pts, self._splined_lensmodel, self._splined_intrinsics)

    def get_intrinsics(self) -> Intrinsics:
        """Return pinhole camera intrinsics (matches the post-transform frame)."""
        return self.intrinsics

    @property
    def height(self):
        """Height of the pinhole-reprojected frame."""
        return self._height

    @property
    def width(self):
        """Width of the pinhole-reprojected frame."""
        return self._width
