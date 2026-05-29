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
        """Load pinhole calibration + undistortion map; defer camera open to load_threaded_cam()."""
        intrinsics_dir = (
            f"{path_to.calibration_presets}/{config.hardware.camera_intrinsics_path}"
        )
        self.intrinsics: Intrinsics = Intrinsics(
            np.load(f"{intrinsics_dir}/dist.pkl", allow_pickle=True),
            np.load(f"{intrinsics_dir}/camera_matrix.pkl", allow_pickle=True),
        )
        # Per-pixel map that reprojects raw frames onto the pinhole model above.
        # Built once by utils/camera_calibration/generate_pinhole_from_mrcal.py.
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
        """Return the most recent frame, reprojected onto the pinhole model."""
        assert self.cap, ("Please run load_threaded_cam before getting frame")

        timestamp = time.perf_counter()
        raw = self.cap.read()
        frame = mrcal.transform_image(raw, self._mapxy)

        return Frame(data=frame, timestamp=timestamp)


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
