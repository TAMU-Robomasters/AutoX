"""Video stream from a standard USB webcam using OpenCV."""

import cv2 as cv
import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream
from src.toolbox.globals import path_to


class WebCamVideoStream(VideoStream):
    """Video stream from a standard USB webcam using OpenCV."""

    def __init__(self, index: int) -> None:
        """Initialize the webcam video stream."""
        self.intrinsics: Intrinsics = Intrinsics(
            np.load(
                f"{path_to.calibration_presets}/circlet_cam_1/dist.pkl",
                allow_pickle=True,
            ),
            np.load(
                f"{path_to.calibration_presets}/circlet_cam_1/camera_matrix.pkl",
                allow_pickle=True,
            ),
        )
        self.cap = cv.VideoCapture(index)  # Use the default webcam
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")

        self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)

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
