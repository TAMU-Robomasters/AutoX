"""Circlet: a ring of body-fixed cameras feeding 360 detections to auto-aim.

Runs ``_CIRCLET_N`` unsynchronized cameras. Each camera gets its **own pinned
worker process** that reads that camera's ``CameraDriver`` ring **zero-copy**
(iceoryx2 -- frames never leave shared memory), runs the classical detector, and
transforms the panels into the **chassis frame** with the camera's static
extrinsic. Workers send the small per-camera ``CircletPanel`` lists back to the
engine over a result queue; the engine aggregates them and publishes a single
:class:`CircletDetections` to the auto-aim engine over the interim pub/sub queue
(``publishes_queue = "circlet_detections"``). Cameras are not synchronized.

WHY PER-CAMERA PROCESSES (vs an in-engine pool): detection is the loop
bottleneck, and a worker pool had to *pickle* every 2.7 MB frame across the
process boundary (a ~34 Hz ceiling). Pinning a worker to each camera aligns the
process split with the existing zero-copy frame boundary, so only tiny panel
data crosses IPC -- true ~Nx parallelism with no frame copies. The single
``circlet_detections`` publisher (this engine) and the auto-aim consumer are
unchanged; only the engine's internals fan out.

LIFECYCLE: workers are forked in :meth:`run` right after logging setup -- before
any iceoryx2 node/thread exists in the engine -- so they inherit the loaded
config (correct intrinsics) cleanly, then each builds its *own* reader in its own
process. ``_die_with_parent`` (PR_SET_PDEATHSIG=SIGKILL) + ``daemon=True`` ensure
no worker outlives the engine, even on an abrupt terminate. The auto-aim engine
subscribes, lifts chassis -> turret with the live gimbal pose, and folds the
detections into targeting; see ``src/subsystems/circlet_support.py`` and
``plans/07-circlet-engine.md``.
"""

import multiprocessing
import os
import queue as _queue
import signal
import time
from typing import Dict, List, Optional

import numpy as np

from src.core.engine import Engine
from src.drivers.video_stream import CameraDriver, _cfg_get, _die_with_parent
from src.subsystems.circlet_support import extrinsic_from_cfg, panel_to_chassis
from src.subsystems.vision.classical_detector.module import ClassicalDetectorModule
from src.toolbox.globals import config
from src.toolbox.logger import configure_child_logging
from src.types.autoaim import FullStateAutoAimContext
from src.types.circlet import CircletDetections, CircletPanel

# The ring is four cameras (config provides per-camera index/backend/extrinsic).
_CIRCLET_N = 4


def _extrinsic_for(camera_id: int) -> np.ndarray:
    """The camera->chassis 4x4 for one ring camera (identity until calibrated)."""
    cams = getattr(config.circlet, "cameras", [])
    cam = cams[camera_id] if camera_id < len(cams) else {}
    yaw_deg = float(_cfg_get(cam, "yaw_deg", 0.0))
    translation = _cfg_get(cam, "translation", (0.0, 0.0, 0.0))
    return extrinsic_from_cfg(yaw_deg, translation)


def _camera_worker(
    camera_id: int,
    conn: dict,
    result_queue: "multiprocessing.Queue",
    stop_event,
    log_queue,
) -> None:
    """Worker process: read one camera zero-copy, detect, push chassis panels back.

    Builds its OWN ``FrameReader`` (so frames stay in shared memory -- only the
    small ``CircletPanel`` list crosses IPC) and one stateless detector. Sends
    ``(camera_id, [CircletPanel], timestamp)`` per fresh frame; an empty list
    still reports liveness. Never raises out of the loop -- a bad frame yields no
    panels, not a dead worker.
    """
    _die_with_parent()  # kernel SIGKILLs this worker if the engine dies
    # Let the engine coordinate shutdown (stop_event / PDEATHSIG); a worker
    # handling SIGINT itself just spews a traceback on Ctrl-C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_child_logging(log_queue)

    # Each worker runs its own detector; without this every worker's OpenCV
    # grabs ALL cores (N_workers * n_cpu threads >> n_cpu), so busy-scene
    # detections thrash on oversubscribed cores. Give each worker a fair,
    # disjoint slice (cpus / ring size) so the ring detects in true parallel.
    import cv2

    cv2.setNumThreads(max(1, (os.cpu_count() or _CIRCLET_N) // _CIRCLET_N))

    reader = CameraDriver.client(conn)  # builds + starts this camera's reader
    ctx = FullStateAutoAimContext()
    detector = ClassicalDetectorModule(ctx)
    detector.initialize()
    extrinsic = _extrinsic_for(camera_id)
    last_seq = -1

    while not stop_event.is_set():
        frame = reader.latest()
        if frame is None or frame.seq == last_seq:
            time.sleep(0.001)  # no new frame yet -- yield instead of busy-spin
            continue
        last_seq = frame.seq

        ctx.frame = frame.data  # zero-copy view, held only for this detect
        ctx.frame_ts = frame.timestamp
        try:
            detector.run()
            panels = ctx.panels or []
        except Exception:  # noqa: BLE001 -- a bad frame must not kill the worker
            panels = []

        out: List[CircletPanel] = []
        for panel in panels:
            if panel.position is None or not np.all(np.isfinite(panel.position)):
                continue
            out.append(
                panel_to_chassis(
                    panel, extrinsic, camera_id=camera_id, timestamp=frame.timestamp
                )
            )
        try:
            result_queue.put_nowait((camera_id, out, frame.timestamp))
        except _queue.Full:
            pass  # engine is behind; drop -- it only ever wants the newest anyway


def _capture_worker(
    camera_id: int,
    conn: dict,
    result_queue: "multiprocessing.Queue",
    stop_event,
    log_queue,
) -> None:
    """Capture-only worker: read one camera zero-copy and report liveness, NO detect.

    Same lifecycle/transport as :func:`_camera_worker` but it never runs the
    detector -- it just pushes an empty panel list per fresh frame so the engine
    counts true capture fps. Used by :class:`CircletCaptureEngine` to isolate the
    camera/USB/IPC path from detection cost.
    """
    _die_with_parent()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_child_logging(log_queue)

    reader = CameraDriver.client(conn)
    last_seq = -1
    while not stop_event.is_set():
        frame = reader.latest()
        if frame is None or frame.seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = frame.seq
        try:
            result_queue.put_nowait((camera_id, [], frame.timestamp))
        except _queue.Full:
            pass


class CircletEngine(Engine[FullStateAutoAimContext]):
    """Ring-of-cameras engine: per-camera worker processes detect; engine aggregates."""

    #: Driver names this engine reads, one worker each. Default = the four ring
    #: cameras (resolved from ``config.circlet.cameras``). Subclasses override to
    #: add cameras (e.g. the gimbal cam via the ``"frames"`` driver).
    _camera_names = tuple(f"cam_{i}" for i in range(_CIRCLET_N))
    drivers = {name: CameraDriver for name in _camera_names}
    publishes_queue = "circlet_detections"

    #: Per-camera worker forked by :meth:`_start_workers`. Subclasses override to
    #: swap behavior (e.g. :class:`CircletCaptureEngine` runs capture-only).
    _worker_target = staticmethod(_camera_worker)
    #: Label used in the per-second throughput log line.
    _throughput_label = "detect"

    def __init__(self, driver_registry: Optional[dict] = None) -> None:
        self.ctx = FullStateAutoAimContext()
        self._workers: List[multiprocessing.Process] = []
        self._result_q: Optional[multiprocessing.Queue] = None
        self._stop_event = None
        # Detection runs in the per-camera workers, not engine-owned modules, so
        # the engine declares no modules (nothing to wire/validate or run here).
        super().__init__(
            modules=[],
            context_type=FullStateAutoAimContext,
            driver_registry=driver_registry,
        )

    # ------------------------------------------------------------------
    # Lifecycle (override run() to fork workers before any iceoryx2/threads)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Engine loop, forking the per-camera workers before reader threads exist.

        Mirrors ``Engine.run`` but (1) starts the workers right after logging setup
        -- before ``_build_driver_handles`` would create FrameReader threads /
        iceoryx2 nodes -- so forked workers inherit the loaded config but no
        threads/IPC, and (2) does NOT build reader handles in the engine itself
        (the workers own the readers). Workers are torn down on exit.
        """
        self.active = True
        configure_child_logging(self._log_queue)
        self._log_pipeline()
        self._start_workers()  # fork now: config inherited, no engine-side IPC yet
        self.initialize()
        try:
            while self.active:
                self.execute()
                self.update()
        finally:
            self._shutdown_workers()

    def _start_workers(self) -> None:
        """Fork one camera-pinned worker per declared camera."""
        self._result_q = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()
        for cam_id, name in enumerate(type(self)._camera_names):
            conn = self._driver_registry[name]
            worker = multiprocessing.Process(
                target=type(self)._worker_target,
                args=(cam_id, conn, self._result_q, self._stop_event, self._log_queue),
                name=f"CircletCam-{name}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _shutdown_workers(self) -> None:
        """Signal workers to stop and reap them (terminate if they linger)."""
        if self._stop_event is not None:
            self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
        if self._result_q is not None:
            self._result_q.close()
            self._result_q.cancel_join_thread()

    def initialize(self) -> None:
        """Set up loop pacing + throughput counters (workers own the cameras)."""
        self._period = 1.0 / float(getattr(config.circlet, "loop_hz", 30))
        self._published = 0
        # one fresh-frame counter per declared camera, reset each ~1s window
        self._frames = [0] * len(type(self)._camera_names)
        self._window_start = time.perf_counter()

    def execute(self) -> None:
        """One tick: drain worker results, aggregate the newest per camera, publish."""
        results = self._drain_results()  # blocks up to _period for the next result

        # Collapse to the newest panel list per camera this tick.
        latest_per_cam: Dict[int, List[CircletPanel]] = {}
        for camera_id, panels, _ts in results:
            latest_per_cam[camera_id] = panels
            self._frames[camera_id] += 1

        detections: List[CircletPanel] = []
        for panels in latest_per_cam.values():
            detections.extend(panels)

        if detections:
            self.publish(
                CircletDetections(panels=detections, timestamp=time.perf_counter())
            )
            self._published += len(detections)

        self._log_throughput()

    def _drain_results(self) -> List[tuple]:
        """Block up to one loop period for a result, then drain the rest non-blocking.

        Blocking (instead of polling + sleeping) paces the loop to worker output
        with low latency and no busy-spin.
        """
        if self._result_q is None:
            return []
        items: List[tuple] = []
        try:
            items.append(self._result_q.get(timeout=self._period))
        except _queue.Empty:
            return items
        try:
            while True:
                items.append(self._result_q.get_nowait())
        except _queue.Empty:
            pass
        return items

    def _log_throughput(self) -> None:
        """Log per-camera detection fps + published-panel rate once per ~1s window."""
        elapsed = time.perf_counter() - self._window_start
        if elapsed < 1.0:
            return
        names = type(self)._camera_names
        fps = " ".join(
            f"{names[i]}={self._frames[i] / elapsed:.0f}" for i in range(len(names))
        )
        self.log.info(
            "circlet: %s fps [%s]; published %.0f panels/s",
            self._throughput_label,
            fps,
            self._published / elapsed,
        )
        self._published = 0
        self._frames = [0] * len(names)
        self._window_start = time.perf_counter()


class CircletCaptureEngine(CircletEngine):
    """Capture-only ring engine: every camera reads frames but NONE detect.

    A diagnostic harness -- same drivers, worker lifecycle, and per-camera fps
    accounting as :class:`CircletEngine`, but the workers run
    :func:`_capture_worker` (no detector), so the logged ``capture fps [...]`` is
    the true per-camera frame throughput the cameras + USB + IPC ring deliver,
    independent of detection cost. Publishes nothing. Run via
    ``run_circlet_capture.py @CIRCLET_ONLY``.
    """

    publishes_queue = None
    _worker_target = staticmethod(_capture_worker)
    _throughput_label = "capture"


class CircletCaptureAllEngine(CircletCaptureEngine):
    """Capture-only harness for ALL cameras: the 4 ring cams + the gimbal cam.

    Same as :class:`CircletCaptureEngine` but it also opens the main/gimbal camera
    via the ``"frames"`` driver (``config.hardware``, same one the auto-aim engine
    uses), so the ``capture fps [...]`` line reports all five cameras at once --
    the full-system USB bandwidth picture with zero detection. The gimbal cam
    needs its ``config.hardware`` camera fields, so run with a robot profile, e.g.
    ``run_circlet_capture.py @SENTRY @CIRCLET_ONLY``.
    """

    _camera_names = tuple(f"cam_{i}" for i in range(_CIRCLET_N)) + ("frames",)
    drivers = {name: CameraDriver for name in _camera_names}
