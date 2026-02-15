"""Simulated video source used for offline testing.

This module provides a lightweight simulated VideoStream implementation that
wraps a `Video` object and optionally supports threaded frame grabbing for
testing code that expects a live camera source.
"""

#TODO: FIXME to work with new codebase
 
from __future__ import annotations

from threading import Thread
from time import sleep
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, VideoStream
from src.toolbox.globals import config, path_to
from src.toolbox.globals import print as gprint

# project imports
from src.toolbox.video_tools import Video

simulation = config.videostream.simulation


class SimulatedVideoStream(VideoStream):
    """Simple simulated VideoStream wrapper around :class:`Video`.

    This class exposes a small subset of behavior used by production code:
    - sequential access using :meth:`frames`
    - optional threaded grabbing via :class:`CameraThreader`
    - intrinsics access via :meth:`get_intrinsics`

    Attributes:
        video_object: The underlying :class:`Video` instance.
        threaded_object: Optional active :class:`CameraThreader` when using threaded grabbing.
        all_frames: Optional cache of all frames when using ``latest_frame`` mode.
    """

    video_object: Video
    threaded_object: Optional["CameraThreader"]
    all_frames: Optional[Tuple[np.ndarray, ...]]
    start_time: Optional[float]
    frame_rate: Optional[float]

    def __init__(self) -> None:
        """Initialize the simulated video stream."""
        self.video_object = Video(path=simulation.input_file)
        self.threaded_object = None
        self.all_frames = None
        self.start_time = None
        self.frame_rate = None

        # Iterator state
        self._frame_iter: Optional[Iterator[Any]] = None
        self._frame_count: int = 0

        if simulation.grab_method == "threaded_frame":
            self.threaded_object = CameraThreader(
                self, update_rate=float(simulation.threaded_update_rate)
            )
        elif simulation.grab_method == "next_frame":
            # No special setup required for sequential access.
            pass
        elif simulation.grab_method == "latest_frame":
            gprint("Loading all frames into RAM for simulated testing")
            self.all_frames = tuple(self.video_object.frames())
            gprint(f"Found {len(self.all_frames)} frames")
            self.start_time = None
            self.frame_rate = float(simulation.assumed_framerate)
        else:
            raise RuntimeError(
                f"simulated VideoStream was created, but config.videostream.simulation.grab_method "
                f"was {simulation.grab_method!r} instead of one of ['next_frame', 'latest_frame', 'threaded_frame']"
            )

    def frames(self, non_threaded: bool = False) -> Optional[Any]:
        """Return the next available frame.

        If ``non_threaded`` is False and the stream was configured with
        ``grab_method == 'threaded_frame'``, return the most recent frame
        produced by the background :class:`CameraThreader`. Otherwise return
        the next frame from the underlying :class:`Video` iterator. Returns
        ``None`` when no more frames are available.

        Args:
            non_threaded: When True, always pull the next frame from the
                underlying video iterator rather than using the threaded cache.

        Returns:
            The next frame (typically a NumPy array) or ``None`` when exhausted.
        """
        # Threaded path: return most recently produced frame.
        if not non_threaded and simulation.grab_method == "threaded_frame":
            return (
                self.threaded_object.frame if self.threaded_object is not None else None
            )

        # Sequential path: iterate the video frames generator.
        if not hasattr(self, "_frame_iter") or self._frame_iter is None:
            self._frame_iter: Iterator[Any] = iter(self.video_object.frames())
            self._frame_count = 0

        try:
            color_frame = next(self._frame_iter)
        except StopIteration:
            return None

        self._frame_count += 1
        return color_frame

    def get_intrinsics(self) -> Intrinsics:
        """Load and return camera intrinsics.

        The method loads the distortion coefficients and camera matrix from
        NumPy files located in ``path_to.calibration_presets``.

        Returns:
            A tuple ``(distortion_coefficients, camera_matrix)``.
        """
        distortion_matrix = np.load(
            f"{path_to.calibration_presets}/dist.pkl", allow_pickle=True
        )
        camera_matrix = np.load(
            f"{path_to.calibration_presets}/camera_matrix.pkl", allow_pickle=True
        )
        return Intrinsics(distortion_matrix, camera_matrix)


class CameraThreader:
    """Background thread that continuously reads frames from a VideoStream.

    The thread stores the last-read frame on ``self.frame`` so consumers can
    read the latest frame without blocking.
    """

    video_stream: VideoStream
    update_rate: float
    frame: Optional[Any]
    stopped: bool
    thread: Thread

    def __init__(self, video_stream: VideoStream, update_rate: float = 0.001) -> None:
        """Initialize and start the background frame-grabbing thread."""
        self.video_stream = video_stream
        self.update_rate = update_rate

        self.frame = None
        self.stopped = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

        self.start()

    def start(self) -> None:
        """Start the background thread."""
        self.stopped = False
        self.thread.start()

    def update(self) -> None:
        """Thread loop: read frames and sleep according to ``update_rate``."""
        while not self.stopped:
            frame = self.video_stream.get_frame()# FIXME
            if frame is None:
                # No more frames available: terminate the loop.
                break
            self.frame = frame
            sleep(self.update_rate)

        self.stopped = True

    def stop(self) -> None:
        """Signal the thread to stop; does not block until the thread has exited."""
        self.stopped = True

    def __del__(self) -> None:
        """Destructor: ensure the thread is stopped and joined."""
        # Join the thread if it's alive so we don't leave dangling threads on
        # interpreter shutdown. Guard against threads that were never started.
        try:
            if hasattr(self, "thread") and self.thread.is_alive():
                self.thread.join(timeout=0.1)
        except (RuntimeError, AttributeError):
            # Don't raise from a destructor when joining threads on interpreter shutdown.
            pass
