"""Camera capture and (optional) zero-copy frame sharing over iceoryx2.

Two layers, so the camera is usable with or without IPC:

- ``CameraSource`` — owns ``cv.VideoCapture`` and reads frames **in-process**.
  No iceoryx2. Single-process engines (e.g. the detection test) use this
  directly; ``read()`` returns a normal writable ``Frame``.
- ``CameraDriver`` (a ``Driver`` process) + ``FrameReader`` — the cross-process
  path. ``CameraDriver`` wraps a ``CameraSource`` (whose reader thread decodes at
  the full sensor rate) and copies the newest frame into a loaned iceoryx2
  shared-memory slot, then publishes it into a fixed-size ring buffer;
  ``FrameReader`` hands consumers **read-only, zero-copy** views. iceoryx2 is
  imported lazily inside these, so importing this module (and ``CameraSource``)
  never requires iceoryx2.

Scope: raw passthrough (lens-correction "normal" mode) only. Lens correction /
intrinsics / depth and non-USB backends are out of scope; the old mrcal-based
variants are archived under ``archive/``.

Validated facts: ``cv.VideoCapture.read(dst)`` writes into the provided buffer on
this camera (same address) -> we point it at shared memory. iceoryx2 subscriber
borrows are read-only (writing segfaults), so reader views are ``writeable=False``.
"""

from __future__ import annotations

import ctypes
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import numpy as np

from src.core.driver import Driver
from src.toolbox.globals import config, path_to
from src.toolbox.logger import get_logger
from src.types.autoaim import Frame

DEFAULT_FRAME_SERVICE = "autox/frames"

_log = get_logger("video_stream")


# ---------------------------------------------------------------------------
# Static camera info (no camera, no IPC) -- importable accessor
# ---------------------------------------------------------------------------


@dataclass
class Intrinsics:
    """Camera intrinsics: OpenCV camera matrix + distortion coefficients."""

    distortion_coefficients: np.ndarray
    camera_matrix: np.ndarray


class _CameraInfo:
    """Static camera info read from config + the cameramodel file.

    Never opens a camera and needs no IPC -- width/height come from config and
    intrinsics are loaded once from the OPENCV12 cameramodel (normal mode). Use
    it anywhere: ``from src.drivers.video_stream import camera_info``.
    """

    def __init__(self) -> None:
        self._intrinsics: Optional[Intrinsics] = None

    @property
    def width(self) -> int:
        return int(config.hardware.camera_width)

    @property
    def height(self) -> int:
        return int(config.hardware.camera_height)

    def intrinsics(self) -> Intrinsics:
        """Load (and cache) intrinsics from ``<presets>/<name>/opencv12.cameramodel``."""
        if self._intrinsics is None:
            import mrcal  # system package; lazy so importing this module is cheap

            intrinsics_dir = (
                f"{path_to.calibration_presets}/{config.hardware.camera_intrinsics_path}"
            )
            model = mrcal.cameramodel(f"{intrinsics_dir}/opencv12.cameramodel")
            _, idata = model.intrinsics()
            fx, fy, cx, cy = (float(x) for x in idata[:4])
            camera_matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            # OPENCV12 trailing 8 values are OpenCV's rational distortion coeffs.
            dist = np.asarray(idata[4:12], dtype=np.float64).reshape(1, -1)
            self._intrinsics = Intrinsics(dist, camera_matrix)
        return self._intrinsics


camera_info = _CameraInfo()

# Service settings must match on publisher and subscriber (open_or_create).
_HISTORY_SIZE = 1
_SUBSCRIBER_BUFFER = 2
_SUBSCRIBER_MAX_BORROWED = 4
_MAX_PUBLISHERS = 1
_MAX_SUBSCRIBERS = 8


def make_frame_payload_type(width: int, height: int) -> Type[ctypes.Structure]:
    """Build the fixed-size iceoryx2 payload type for a width x height BGR frame.

    Built at runtime (not import) so nothing depends on a camera being open and
    the size is pinned to the static configured resolution.
    """
    n_bytes = width * height * 3

    class FramePayload(ctypes.Structure):
        _fields_ = [
            ("timestamp_ns", ctypes.c_uint64),
            ("seq", ctypes.c_uint64),
            ("width", ctypes.c_uint32),
            ("height", ctypes.c_uint32),
            ("pixels", ctypes.c_uint8 * n_bytes),
        ]

    return FramePayload


def _die_with_parent() -> None:
    """preexec_fn: ask the kernel to SIGKILL this child if its parent dies.

    Ensures the ffmpeg subprocess never outlives (and keeps holding the camera
    after) the driver process, even on an abrupt terminate()/crash.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


class CameraSource:
    """In-process camera capture via an ffmpeg subprocess (no IPC).

    ffmpeg captures MJPEG and decodes to raw BGR, multi-threading the JPEG decode
    that OpenCV's built-in ``VideoCapture`` does single-threaded -- so it reaches
    the camera's full ~90fps ceiling instead of OpenCV's ~72. A daemon thread
    reads decoded frames off ffmpeg's stdout pipe and keeps the newest one;
    consumers never block the capture. Manual exposure is applied via ``v4l2-ctl``
    because ffmpeg's v4l2 input doesn't expose camera controls.
    """

    def __init__(self, camera_index: Optional[int] = None) -> None:
        self._camera_index = camera_index
        self._proc: Optional[subprocess.Popen] = None
        self.width = 0
        self.height = 0
        self._frame_bytes = 0
        self._cond = threading.Condition()
        self._latest: Optional[Tuple[np.ndarray, float, int]] = None
        self._seq = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _device(self) -> str:
        index = (
            self._camera_index
            if self._camera_index is not None
            else int(config.hardware.camera_index)
        )
        return f"/dev/video{index}"

    def _desired_controls(self) -> "dict[str, int]":
        """Resolve the full set of v4l2 controls to apply, from config.

        Two config sources, merged in this order (later overrides earlier):

        - ``hardware.camera_exposure`` (legacy shorthand, also read by the
          OpenCV streaming paths): expands to manual exposure
          (``auto_exposure=1`` + ``exposure_time_absolute``).
        - ``hardware.camera_controls`` (a mapping of ``v4l2-ctl --set-ctrl``
          names to values, e.g. ``brightness``, ``contrast``, ``gain``):
          applied verbatim. This is the authoritative, tunable control set --
          edit it with ``utils/tune_camera.py``.

        Insertion order is preserved, so put any "enable" toggle before the
        value it gates (e.g. ``white_balance_automatic`` before
        ``white_balance_temperature``).
        """
        controls: "dict[str, int]" = {}
        exposure = getattr(config.hardware, "camera_exposure", None)
        if exposure is not None:
            controls["auto_exposure"] = 1  # 1 = Manual Mode
            controls["exposure_time_absolute"] = int(exposure)
        extra = getattr(config.hardware, "camera_controls", None)
        if extra:
            for name, value in dict(extra).items():
                if value is not None:
                    controls[name] = int(value)
        return controls

    def _apply_controls(self, device: str) -> None:
        """Apply each configured v4l2 control via ``v4l2-ctl --set-ctrl``.

        Controls are set one at a time so an unsupported/invalid control on a
        given camera is skipped (logged at debug) instead of aborting the rest.
        """
        for name, value in self._desired_controls().items():
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                _log.debug(
                    "skipped v4l2 control %s=%s on %s: %s",
                    name, value, device,
                    result.stderr.decode(errors="replace").strip(),
                )

    def open(self) -> None:
        """Start ffmpeg capture + the background reader thread."""
        self.width = int(config.hardware.camera_width)
        self.height = int(config.hardware.camera_height)
        self._frame_bytes = self.width * self.height * 3
        device = self._device()
        fps = int(getattr(config.hardware, "camera_fps", None) or 90)

        self._apply_controls(device)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-framerate", str(fps), "-video_size", f"{self.width}x{self.height}",
            "-i", device,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes,
            preexec_fn=_die_with_parent,  # ffmpeg dies if this process is killed
        )
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read exactly one BGR frame off the pipe (handles partial reads)."""
        buf = np.empty(self._frame_bytes, dtype=np.uint8)
        view = memoryview(buf)
        got = 0
        stdout = self._proc.stdout  # type: ignore[union-attr]
        while got < self._frame_bytes:
            n = stdout.readinto(view[got:])
            if not n:  # EOF -> ffmpeg exited
                return None
            got += n
        return buf.reshape(self.height, self.width, 3)

    def _reader(self) -> None:
        """Background loop: always hold the newest decoded frame."""
        while self._running:
            frame = self._read_frame()
            if frame is None:
                break
            ts = time.perf_counter()
            with self._cond:
                self._seq += 1
                self._latest = (frame, ts, self._seq)
                self._cond.notify_all()

    def _wait_new(self, last_seq: int, timeout: float):
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
        """Block for a frame newer than ``last_seq``, copy it into ``dst``; return (ts, seq).

        The decode already happened on the reader thread; this is a single
        copy of the newest frame into ``dst`` (e.g. a shared-memory slot).
        """
        latest = self._wait_new(last_seq, timeout)
        if latest is None:
            return None
        data, ts, seq = latest
        np.copyto(dst, data)
        return ts, seq

    def close(self) -> None:
        self._running = False
        if self._proc is not None:
            self._proc.kill()  # makes the reader's pipe read hit EOF
            self._proc.wait()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _open_service(node, payload_type: Type[ctypes.Structure], service_name: str):
    """Open (or create) the shared frame service with the agreed ring-buffer config."""
    import iceoryx2 as iox2  # type: ignore[import-untyped]

    return (
        node.service_builder(iox2.ServiceName.new(service_name))
        .publish_subscribe(payload_type)
        .enable_safe_overflow(True)  # publisher overwrites oldest, never blocks
        .history_size(_HISTORY_SIZE)
        .subscriber_max_buffer_size(_SUBSCRIBER_BUFFER)
        .subscriber_max_borrowed_samples(_SUBSCRIBER_MAX_BORROWED)
        .max_publishers(_MAX_PUBLISHERS)
        .max_subscribers(_MAX_SUBSCRIBERS)
        .open_or_create()
    )


class CameraDriver(Driver):
    """Owns the camera and decodes frames straight into a shared-memory ring buffer."""

    def __init__(
        self,
        service_name: str = DEFAULT_FRAME_SERVICE,
        requires: str = "camera",
        camera_index: Optional[int] = None,
    ) -> None:
        super().__init__(requires=requires)
        self._service_name = service_name
        self._camera_index = camera_index

    # -- orchestrator factory contract (provision in parent, client in child) --

    @staticmethod
    def provision(name: str) -> dict:
        """Parent-side: allocate this driver's transport (here: a service name)."""
        return {"service": f"autox/{name}"}

    @classmethod
    def from_conn(cls, conn: dict) -> "CameraDriver":
        """Build the driver process from its provisioned connection info."""
        return cls(service_name=conn["service"])

    @staticmethod
    def client(conn: dict) -> "FrameReader":
        """Child-side: build + start a consumer handle (a FrameReader)."""
        reader = FrameReader(conn["service"])
        reader.start()
        return reader

    def initialize(self) -> None:  # runs in the child process
        import iceoryx2 as iox2  # type: ignore[import-untyped]

        self._cam = CameraSource(self._camera_index)
        self._cam.open()
        self._width, self._height = self._cam.width, self._cam.height
        self._payload_type = make_frame_payload_type(self._width, self._height)

        self._node = (
            iox2.NodeBuilder.new()
            .name(iox2.NodeName.new("autox_camera_driver"))
            .create(iox2.ServiceType.Ipc)
        )
        self._service = _open_service(
            self._node, self._payload_type, self._service_name
        )
        self._publisher = self._service.publisher_builder().create()
        self._last_seq = -1

    def execute(self) -> None:
        sample = self._publisher.loan_uninit()
        payload = sample.payload().contents  # writable shared-memory slot
        dst = np.ctypeslib.as_array(payload.pixels).reshape(
            self._height, self._width, 3
        )
        # Block for the newest frame (decoded on the reader thread) and copy it
        # into the loaned slot. One copy; the decode overlapped the last publish.
        result = self._cam.read_into(dst, last_seq=self._last_seq)
        if result is None:
            return  # no new frame within timeout; drop the loan
        timestamp, seq = result
        self._last_seq = seq
        payload.timestamp_ns = int(timestamp * 1e9)
        payload.seq = seq
        payload.width = self._width
        payload.height = self._height
        sample.assume_init().send()


class FrameReader:
    """Subscriber handle: hands out read-only zero-copy frames, non-blocking."""

    def __init__(
        self,
        service_name: str = DEFAULT_FRAME_SERVICE,
        node_name: str = "autox_frame_reader",
    ) -> None:
        self._service_name = service_name
        self._node_name = node_name
        self._started = False

    def start(self) -> None:  # call in the consuming process
        import iceoryx2 as iox2  # type: ignore[import-untyped]

        self._width = int(config.hardware.camera_width)
        self._height = int(config.hardware.camera_height)
        self._payload_type = make_frame_payload_type(self._width, self._height)
        self._node = (
            iox2.NodeBuilder.new()
            .name(iox2.NodeName.new(self._node_name))
            .create(iox2.ServiceType.Ipc)
        )
        self._service = _open_service(
            self._node, self._payload_type, self._service_name
        )
        self._subscriber = self._service.subscriber_builder().create()
        self._started = True

    def latest(self) -> Optional[Frame]:
        """Return the newest available frame as a read-only view, or None.

        Drains the subscriber queue so a slow consumer always jumps to the most
        recent frame rather than walking a backlog. The returned frame's ``data``
        is read-only (``writeable=False``); ``.copy()`` it before annotating.
        """
        if not self._started:
            self.start()

        sample = None
        while True:
            received = self._subscriber.receive()
            if received is None:
                break
            sample = received  # keep only the newest

        if sample is None:
            return None

        payload = sample.payload().contents
        view = np.ctypeslib.as_array(payload.pixels).reshape(
            self._height, self._width, 3
        )
        view.flags.writeable = False  # turn an accidental write into a clean error
        frame = Frame(
            data=view,
            timestamp=payload.timestamp_ns / 1e9,
            seq=int(payload.seq),
        )
        frame._sample = sample  # keep the borrow alive for the view's lifetime
        return frame
