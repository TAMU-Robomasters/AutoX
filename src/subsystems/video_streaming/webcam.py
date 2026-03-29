"""Video stream from a standard USB webcam using OpenCV."""

import cv2 as cv
import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream
from src.toolbox.globals import path_to, config


class WebCamVideoStream(VideoStream):
    """Video stream from a standard USB webcam using OpenCV."""

    def __init__(self, index: int) -> None:
        """Initialize the webcam video stream."""
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
        self.cap = cv.VideoCapture(index)  # Use the default webcam
                
        # TODO get rid of this and actually fix threading multiprocessing bug
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, config.hardware.camera_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.hardware.camera_height)
        self.cap.set(cv.CAP_PROP_EXPOSURE, config.hardware.camera_exposure)
        self.cap.set(cv.CAP_PROP_FPS, config.hardware.camera_fps)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")

    def get_frame(self):
        """Return the most recent frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read frame from webcam.")
        return frame

    def get_intrinsics(self) -> Intrinsics:
        """Return camera intrinsics loaded from calibration presets."""
        return self.intrinsics

    @property
    def height(self) -> int:  # noqa: D102
        return int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:  # noqa: D102
        return int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
