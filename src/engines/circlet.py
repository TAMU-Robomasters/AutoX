"""Circlet: a ring of body-fixed cameras feeding 360 detections to auto-aim.

Runs ``_CIRCLET_N`` unsynchronized cameras (each its own shared ``CameraDriver``
at ~15 fps), pushes every fresh frame through the classical detector, transforms
the panels into the **chassis frame** with that camera's static extrinsic, and
publishes a :class:`CircletDetections` to the auto-aim engine over the interim
pub/sub queue (``publishes_queue = "circlet_detections"``). The cameras are
intentionally not synchronized -- each is handled independently per loop.

The auto-aim engine subscribes, lifts chassis -> turret with the live gimbal
pose, and folds the detections into its classification/targeting decision so the
turret can react to robots its gimbal camera can't see. See
``src/subsystems/circlet_support.py`` for the (detector-free) geometry helpers
and ``plans/07-circlet-engine.md`` for the design.
"""

import time
from typing import List, Optional

import numpy as np

from src.core.engine import Engine
from src.core.module import Module
from src.drivers.video_stream import CameraDriver, _cfg_get
from src.subsystems.circlet_support import extrinsic_from_cfg, panel_to_chassis
from src.subsystems.vision.classical_detector.module import ClassicalDetectorModule
from src.toolbox.globals import config
from src.types.autoaim import FullStateAutoAimContext
from src.types.circlet import CircletDetections, CircletPanel

# The ring is four cameras (config provides per-camera index/backend/extrinsic).
_CIRCLET_N = 4


class CircletEngine(Engine[FullStateAutoAimContext]):
    """Ring-of-cameras engine: detect per camera, publish chassis-frame panels."""

    drivers = {f"cam_{i}": CameraDriver for i in range(_CIRCLET_N)}
    publishes_queue = "circlet_detections"

    def __init__(self, driver_registry: Optional[dict] = None) -> None:
        # One context + detector per camera (the detector is stateless: reads
        # ctx.frame, writes ctx.panels), so the cameras don't share frame state.
        self._contexts = [FullStateAutoAimContext() for _ in range(_CIRCLET_N)]
        self._detectors: List[Module] = [
            ClassicalDetectorModule(c) for c in self._contexts
        ]
        self.ctx = self._contexts[0]
        super().__init__(
            modules=self._detectors,
            context_type=FullStateAutoAimContext,
            driver_registry=driver_registry,
        )

    def initialize(self) -> None:
        """Build the per-camera FrameReader handles, extrinsics, and loop pacing."""
        self._readers = [self.driver(f"cam_{i}") for i in range(_CIRCLET_N)]
        self._last_seq = [-1] * _CIRCLET_N
        self._extrinsics = self._load_extrinsics()
        self._period = 1.0 / float(getattr(config.circlet, "loop_hz", 30))
        self._published = 0
        self._window_start = time.perf_counter()

    def _load_extrinsics(self) -> List[np.ndarray]:
        """One camera->chassis 4x4 per ring camera (identity until calibrated)."""
        cams = getattr(config.circlet, "cameras", [])
        extrinsics = []
        for i in range(_CIRCLET_N):
            cam = cams[i] if i < len(cams) else {}
            yaw_deg = float(_cfg_get(cam, "yaw_deg", 0.0))
            translation = _cfg_get(cam, "translation", (0.0, 0.0, 0.0))
            extrinsics.append(extrinsic_from_cfg(yaw_deg, translation))
        return extrinsics

    def execute(self) -> None:
        """One tick: detect on each camera with a fresh frame, publish the union."""
        start = time.perf_counter()
        detections: List[CircletPanel] = []

        for i in range(_CIRCLET_N):
            frame = self._readers[i].latest()
            if frame is None or frame.seq == self._last_seq[i]:
                continue  # no new frame from this (unsynchronized) camera
            self._last_seq[i] = frame.seq

            ctx = self._contexts[i]
            ctx.frame = frame.data
            ctx.frame_ts = frame.timestamp
            self._detectors[i].run()

            panels = ctx.panels
            if not panels:
                continue
            extrinsic = self._extrinsics[i]
            for panel in panels:
                if panel.position is None or not np.all(np.isfinite(panel.position)):
                    continue
                detections.append(
                    panel_to_chassis(
                        panel, extrinsic, camera_id=i, timestamp=frame.timestamp
                    )
                )

        if detections:
            self.publish(CircletDetections(panels=detections, timestamp=start))
            self._published += len(detections)

        self._log_throughput()
        self._pace(start)

    def _pace(self, start: float) -> None:
        """Sleep only enough to keep the loop under ``circlet.loop_hz``."""
        remaining = self._period - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    def _log_throughput(self) -> None:
        """Log published-panel rate once per ~1s window."""
        elapsed = time.perf_counter() - self._window_start
        if elapsed < 1.0:
            return
        self.log.info("circlet: published %.0f panels/s", self._published / elapsed)
        self._published = 0
        self._window_start = time.perf_counter()
