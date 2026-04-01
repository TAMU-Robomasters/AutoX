"""USB camera video stream subsystem."""

from __future__ import annotations

import queue
import threading

import cv2 as cv
import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream

# project imports
from src.toolbox.globals import path_to, print, config


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
        """Load calibration and start the capture pipeline."""
        self.intrinsics: Intrinsics = Intrinsics(
            np.load(
                f"{path_to.calibration_presets}/main_sentry_cam/dist.pkl",
                allow_pickle=True,
            ),
            np.load(
                f"{path_to.calibration_presets}/main_sentry_cam/camera_matrix.pkl",
                allow_pickle=True,
            ),
        )

        try:
            self.cap = BufferlesCvCapture(index)
        except Exception as e:
            print(f"Error in BufferlesCvCapture: {e}")
            raise e

    def get_frame(self):
        """Return the most recent frame from the USB camera."""
        return self.cap.read()

    def get_intrinsics(self) -> Intrinsics:
        """Return camera intrinsics loaded from calibration presets."""
        return self.intrinsics

    # TODO: actually implement these
    @property
    def height(self):
        """Get height."""
        return int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self):
        """Get width."""
        return int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
