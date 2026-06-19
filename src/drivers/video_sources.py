"""Pluggable capture backends for :class:`~src.drivers.video_stream.CameraDriver`.

A *source* owns frame production for ONE camera and exposes the small surface
``CameraDriver`` needs -- the same one the legacy ffmpeg ``CameraSource`` exposes:

    open()                      -> start producing frames
    read(last_seq, timeout)     -> Optional[Frame]
    read_into(dst, last_seq, timeout) -> Optional[(timestamp, seq)]
    close()
    .width / .height            -> int

Three backends, picked by name via :func:`make_source` (``config.hardware.
video_backend`` for the main cam, or per-camera for circlet):

- ``pyav``   -- :class:`PyAvCameraSource`: in-process libav (PyAV) MJPEG decode,
  multi-threaded like the old ffmpeg subprocess but with no shell; V4L2 controls
  (exposure / gain / fps) via ``v4l2-python3`` ioctls. ``av`` and ``v4l2`` are
  **lazy-imported inside the class** (the optional ``linux`` extra) so importing
  this module never requires them.
- ``mock``   -- :class:`MockVideoSource`: loops a configurable video file, resized
  to the configured resolution and paced at a (runtime-variable) fps. Falls back
  to synthetic frames when the file is missing or an un-pulled git-lfs pointer.
- ``ffmpeg`` -- the original subprocess capture (kept for parity; lazily pulled
  from ``video_stream`` to avoid an import cycle).

All three share :class:`_ThreadedFrameSource`: a daemon reader thread that always
holds the newest decoded frame so consumers never block the producer.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Tuple

import cv2 as cv
import numpy as np

from src.toolbox.logger import get_logger
from src.types.autoaim import Frame

log = get_logger("video_sources")

# Repo root (…/AutoX), so a relative mock_video_path resolves regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class FrameSource(Protocol):
    """The capture-source surface ``CameraDriver`` depends on.

    Both :class:`_ThreadedFrameSource` (pyav / mock) and the legacy ffmpeg
    ``CameraSource`` satisfy it structurally.
    """

    width: int
    height: int

    def open(self) -> None:
        """Start producing frames."""

    def read(self, last_seq: int = ..., timeout: float = ...) -> Optional[Frame]:
        """Return the newest frame after ``last_seq`` as a ``Frame``, or None."""

    def read_into(
        self, dst: np.ndarray, last_seq: int = ..., timeout: float = ...
    ) -> Optional[Tuple[float, int]]:
        """Copy the newest frame after ``last_seq`` into ``dst``; return (ts, seq) or None."""

    def close(self) -> None:
        """Release capture resources."""


class _ThreadedFrameSource:
    """Base: a background thread keeps the newest frame; consumers never block.

    Subclasses implement :meth:`_frames` (a generator of BGR ``uint8`` ndarrays);
    the base handles sequencing, the newest-frame condition variable, and the
    ``read`` / ``read_into`` surface ``CameraDriver`` consumes.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self._cond = threading.Condition()
        self._latest: Optional[Tuple[np.ndarray, float, int]] = None
        self._seq = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # -- subclass hooks ----------------------------------------------------
    def _frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames forever (or until exhausted). Implemented by subclasses."""
        raise NotImplementedError

    def _on_close(self) -> None:
        """Release backend resources (container / capture). No-op default."""

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        """Start the background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        try:
            for frame in self._frames():
                if not self._running:
                    break
                ts = time.perf_counter()
                with self._cond:
                    self._seq += 1
                    self._latest = (frame, ts, self._seq)
                    self._cond.notify_all()
        except Exception:  # keep a backend crash from killing the process silently
            log.exception("%s reader thread died", type(self).__name__)
        finally:
            self._running = False

    def _wait_new(
        self, last_seq: int, timeout: float
    ) -> Optional[Tuple[np.ndarray, float, int]]:
        with self._cond:
            got = self._cond.wait_for(
                lambda: self._latest is not None and self._latest[2] != last_seq,
                timeout=timeout,
            )
            return self._latest if got else None

    def read(self, last_seq: int = -1, timeout: float = 1.0) -> Optional[Frame]:
        """Block until a frame newer than ``last_seq``; return it as a ``Frame``."""
        latest = self._wait_new(last_seq, timeout)
        if latest is None:
            return None
        data, ts, seq = latest
        return Frame(data=data, timestamp=ts, seq=seq)

    def read_into(
        self, dst: np.ndarray, last_seq: int = -1, timeout: float = 1.0
    ) -> Optional[Tuple[float, int]]:
        """Block for a frame newer than ``last_seq``, copy it into ``dst``; return (ts, seq)."""
        latest = self._wait_new(last_seq, timeout)
        if latest is None:
            return None
        data, ts, seq = latest
        np.copyto(dst, data)
        return ts, seq

    def close(self) -> None:
        self._running = False
        self._on_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# ---------------------------------------------------------------------------
# Mock: loop a video file (synthetic fallback), resized + paced to config
# ---------------------------------------------------------------------------


def _is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an un-pulled git-lfs pointer (tiny text stub)."""
    try:
        if path.stat().st_size > 4096:
            return False
        with path.open("rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs.github.com")
    except OSError:
        return False


class MockVideoSource(_ThreadedFrameSource):
    """Loops a video file at a (runtime-variable) fps; synthetic frames as fallback.

    Every yielded frame is resized to ``width x height`` so the iceoryx2 payload
    matches the configured resolution. ``target_fps`` is a plain attribute -- set
    it live to change the rate at runtime (``fps_jitter`` adds per-frame noise so
    the stream isn't perfectly periodic, mimicking a real camera).
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        video_path: Optional[str] = None,
        fps_jitter: float = 0.0,
    ) -> None:
        super().__init__(width, height)
        self.target_fps = float(fps) if fps and fps > 0 else 30.0
        self.fps_jitter = float(fps_jitter)
        self._video_path = self._resolve(video_path) if video_path else None

    @staticmethod
    def _resolve(video_path: str) -> Path:
        p = Path(video_path)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    def _use_synthetic(self) -> bool:
        p = self._video_path
        if p is None or not p.exists() or _is_lfs_pointer(p):
            return True
        cap = cv.VideoCapture(str(p))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return not ok

    def _sleep_for_fps(self, start: float) -> None:
        fps = max(self.target_fps, 1e-3)
        interval = 1.0 / fps
        if self.fps_jitter:
            interval *= 1.0 + np.random.uniform(-self.fps_jitter, self.fps_jitter)
        remaining = interval - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    def _frames(self) -> Iterator[np.ndarray]:
        if self._use_synthetic():
            log.warning(
                "MockVideoSource: %s missing/unreadable (git-lfs pointer?) -> "
                "synthetic frames. Run `git lfs pull` for real footage.",
                self._video_path,
            )
            yield from self._synthetic_frames()
            return
        log.info(
            "MockVideoSource looping %s @ ~%.0f fps", self._video_path, self.target_fps
        )
        yield from self._video_frames()

    def _video_frames(self) -> Iterator[np.ndarray]:
        cap = cv.VideoCapture(str(self._video_path))
        try:
            while self._running:
                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:  # loop back to the start
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv.resize(frame, (self.width, self.height))
                yield np.ascontiguousarray(frame)
                self._sleep_for_fps(start)
        finally:
            cap.release()

    def _synthetic_frames(self) -> Iterator[np.ndarray]:
        """A moving gradient + a sliding bright bar, so frames change each seq."""
        h, w = self.height, self.width
        base_y = np.linspace(0, 120, h, dtype=np.uint8)[:, None]
        base = np.repeat(base_y, w, axis=1)
        i = 0
        while self._running:
            start = time.perf_counter()
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            shift = (i * 7) % w
            frame[:, :, 0] = np.roll(base, shift, axis=1)  # B
            frame[:, :, 2] = np.roll(base, -shift, axis=1)  # R
            x0 = (i * 11) % max(w - 60, 1)
            frame[h // 2 - 30 : h // 2 + 30, x0 : x0 + 60] = (255, 255, 255)
            yield frame
            i += 1
            self._sleep_for_fps(start)


# ---------------------------------------------------------------------------
# PyAV: in-process libav capture + v4l2-python3 controls (the `linux` extra)
# ---------------------------------------------------------------------------


class PyAvCameraSource(_ThreadedFrameSource):
    """V4L2 capture via PyAV (libav), with V4L2 controls via ``v4l2-python3``.

    Replaces the ffmpeg subprocess: PyAV decodes MJPEG in-process with libav's
    multi-threaded decoder (``thread_type='AUTO'``), reaching the camera's full
    rate without OpenCV's single-threaded-decode ceiling. ``av`` and ``v4l2`` are
    imported lazily so this module imports without the ``linux`` extra installed.
    """

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int = 90,
        exposure: Optional[int] = None,
        input_format: str = "mjpeg",
    ) -> None:
        super().__init__(width, height)
        self.index = int(index)
        self.fps = int(fps)
        self.exposure = exposure
        self.input_format = input_format
        self._container: Optional[Any] = None  # av.container.InputContainer (lazy)

    @property
    def device(self) -> str:
        """The ``/dev/videoN`` path for this camera index."""
        return f"/dev/video{self.index}"

    def _apply_controls(self) -> None:
        """Set manual exposure via v4l2-python3 ioctls; fall back to v4l2-ctl."""
        if self.exposure is None:
            return
        try:
            import fcntl

            import v4l2  # type: ignore[import-untyped]

            with open(self.device, "rb+", buffering=0) as fd:
                for cid, value in (
                    (v4l2.V4L2_CID_EXPOSURE_AUTO, v4l2.V4L2_EXPOSURE_MANUAL),  # =1
                    (v4l2.V4L2_CID_EXPOSURE_ABSOLUTE, int(self.exposure)),
                ):
                    ctrl = v4l2.v4l2_control()
                    ctrl.id = cid
                    ctrl.value = value
                    fcntl.ioctl(fd, v4l2.VIDIOC_S_CTRL, ctrl)
        except Exception as e:  # missing v4l2 / unsupported control -> subprocess
            log.warning("v4l2-python3 control failed (%s); falling back to v4l2-ctl", e)
            import subprocess

            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    self.device,
                    "--set-ctrl",
                    "auto_exposure=1",
                    "--set-ctrl",
                    f"exposure_time_absolute={int(self.exposure)}",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _frames(self) -> Iterator[np.ndarray]:
        import av  # type: ignore[import-untyped]  # the `linux` extra

        self._apply_controls()  # controls must be set before/while the device is open
        self._container = av.open(
            self.device,
            format="v4l2",
            options={
                "input_format": self.input_format,
                "video_size": f"{self.width}x{self.height}",
                "framerate": str(self.fps),
            },
        )
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"  # multi-threaded decode (the ffmpeg-parity win)
        for frame in self._container.decode(stream):
            if not self._running:
                break
            yield frame.to_ndarray(format="bgr24")

    def _on_close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_source(
    backend: str,
    *,
    index: int,
    width: int,
    height: int,
    fps: int,
    exposure: Optional[int] = None,
    mock_video_path: Optional[str] = None,
    mock_fps_jitter: float = 0.0,
) -> FrameSource:
    """Build the capture source named by ``backend`` (pyav | mock | ffmpeg)."""
    backend = (backend or "pyav").lower()
    if backend == "mock":
        return MockVideoSource(
            width,
            height,
            fps=fps,
            video_path=mock_video_path,
            fps_jitter=mock_fps_jitter,
        )
    if backend == "pyav":
        return PyAvCameraSource(index, width, height, fps=fps, exposure=exposure)
    if backend == "ffmpeg":
        # Lazy import to avoid a cycle (video_stream imports this module's factory).
        # The legacy ffmpeg source is config-driven (reads the global hardware
        # block in open()); per-camera width/height aren't threaded through it.
        from src.drivers.video_stream import CameraSource

        return CameraSource(index)
    raise ValueError(f"unknown video backend {backend!r} (expected pyav|mock|ffmpeg)")
