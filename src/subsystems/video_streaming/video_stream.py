"""Video streaming singleton loaded based on config."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.toolbox.globals import config


@dataclass
class Intrinsics:
    """Camera intrinsics."""

    distortion_coefficients: np.ndarray
    camera_matrix: np.ndarray


class VideoStream(ABC):
    """Singleton class for video streaming subsystem."""

    supports_depth: bool = False
    supports_intrinsics: bool = False

    @property
    @abstractmethod
    def height(self) -> int:
        """Get height."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Get width."""

    @property
    def center(self) -> Tuple[int, int]:
        """Get u,v pixel center."""
        return (self.width // 2, self.height // 2)

    @abstractmethod
    def get_frame(self):
        """Return the most recent frame from the video stream."""

    def get_depth_at(self, u: int, v: int) -> Optional[float]:
        """Return the depth at pixel (u, v) in meters."""
        raise NotImplementedError("This video stream does not have a depth channel.")

    def get_xyz_at(self, u: int, v: int, depth: float) -> tuple[float, float, float]:
        """Return the (X, Y, Z) coordinates at pixel (u, v) in meters."""
        raise NotImplementedError("This video stream does not have a depth channel.")

    def get_pixel_at(self, x: float, y: float, z: float) -> tuple[int, int]:
        """Return the (u, v) pixel coordinates for the given (X, Y, Z) in meters."""
        raise NotImplementedError(
            "This video stream does not have a depth channel or provide intrinsics."
        )

    def get_intrinsics(self) -> Intrinsics:
        """Return the camera intrinsics."""
        raise NotImplementedError("This video stream does not provide intrinsics.")


def create_video_stream() -> VideoStream:
    """Create and return a VideoStream instance based on config.yaml."""
    if config.hardware.camera == "realsense":
        from src.subsystems.video_streaming.realsense import RealSenseVideoStream

        return RealSenseVideoStream()
    elif config.hardware.camera == "usb_cam":
        from src.subsystems.video_streaming.usb_cam import USBCamVideoStream

        return USBCamVideoStream(config.hardware.camera_index)
    elif config.hardware.camera == "webcam":
        from src.subsystems.video_streaming.webcam import WebCamVideoStream

        return WebCamVideoStream(config.hardware.camera_index)
    else:
        raise ValueError(f"Unsupported camera type: {config.hardware.camera}")


video_stream: VideoStream = create_video_stream()
