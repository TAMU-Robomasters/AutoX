"""Auto-aim engine for a single, fixed-radius, constant-rate spinning target.

The aerial rig is a pole spinning at a known ~10 RPM with 3 identical panels
120deg apart at the same height, on arms of equal radius. There is exactly one
target, so no classification / targeting / state machine: whatever panels the
detector returns are *it*. Because the spin is slow (below the shot-timing
threshold) we continuously fire the whole time.

Pipeline each tick:
    1. ClassicalDetector                  -> panels
    2. (execute logic)                    -> transform panels to turret frame
    3. EkfEstimation                      -> estimate (centre + spin + height)
    4. SpinningTargetContinuousFire       -> solution
    5. (update logic)                     -> send solution to embedded

Frame pacing / coast (mirrors ``full_state_autoaim``): the loop runs as fast as
``loop_hz`` allows and does NOT block on the camera. A fresh frame drives the
full vision pipeline; every other tick coasts -- the estimator predicts forward
(``update_with_no_observation``) so pitch/yaw keep flowing between frames. The
engine holds the last turret-frame panels in ``ctx.panels`` while the target is
locked (so the estimator keeps predicting) and nulls them only after
``target_reset_timeout`` with no observation, which resets the estimator so it
re-seeds cleanly on re-acquisition.
"""

from __future__ import annotations

import time
from multiprocessing import Queue
from typing import List, Optional

import numpy as np

from src.core.engine import Engine
from src.drivers.mcu import McuDriver
from src.drivers.video_stream import CameraDriver
from src.subsystems.display import display
from src.subsystems.ekf_estimation import (
    EkfEstimationModule,
    _default_ekf_tracker,
    _default_height_kf,
)
from src.subsystems.embedded_communicator import CVState
from src.subsystems.spinning_target_continuous_fire import (
    SpinningTargetContinuousFireModule,
)
from src.subsystems.turret_frame import transform_panels_to_turret_frame
from src.subsystems.vision import ClassicalDetectorModule
from src.toolbox.globals import config
from src.toolbox.timeout import Timeout
from src.types.autoaim import ArmorPanel, EkfAutoAimContext


class SpinningTargetAutoAimEngine(Engine[EkfAutoAimContext]):
    """Auto-aim engine that tracks one circular spinning target with an EKF."""

    drivers = {"frames": CameraDriver, "mcu": McuDriver}

    def __init__(
        self,
        driver_registry: Optional[dict] = None,
        queue: Optional[Queue] = None,
    ) -> None:
        self._queue = queue
        self.ctx = EkfAutoAimContext()

        self.detection = ClassicalDetectorModule(self.ctx)
        self.estimation = EkfEstimationModule(self.ctx)
        self.continuous_fire = SpinningTargetContinuousFireModule(self.ctx)

        super().__init__(
            modules=[
                self.detection,
                self.estimation,
                self.continuous_fire,
            ],
            context_type=EkfAutoAimContext,
            driver_registry=driver_registry,
        )

    # ------------------------------------------------------------------
    # Child-process resource creation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the EKF + height KF, attach them, open drivers."""
        self.estimation.set_estimator(_default_ekf_tracker())
        self.estimation.set_height_kf(_default_height_kf())

        cfg = config.autoaim
        self.target_timeout = Timeout(duration=cfg.target_reset_timeout_ms / 1e3)
        # Loop runs as fast as it can with this period as a CAP.
        self._period = 1.0 / float(cfg.loop_hz)
        self._n_panels = int(config.spinning_target.n_panels)

        self.mcu = self.driver("mcu")
        self.frames = self.driver("frames")
        self._last_seq = -1

        # Last turret-frame panels; held in ctx.panels while locked so the
        # estimator keeps predicting on coast ticks. None == no target.
        self._locked_panels: Optional[List[ArmorPanel]] = None

        # Fallbacks sent while we don't yet have a solution.
        self._last_pitch: float = float(np.deg2rad(-10))
        self._last_yaw: float = 0.0
        self.alignment_time_ms: int = 255
        self.cv_state: int = CVState.NO_TARGET.value

        # Two FPS tallies: fresh observations vs predict/re-send ticks.
        self._obs_count = 0
        self._no_obs_count = 0
        self._fps_window_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """One tick: process a fresh frame if present, else coast; then solve + stage."""
        start = time.perf_counter()
        self.ctx.start_loop_time = start
        self.ctx.solution = None  # never reuse a stale solution
        self.alignment_time_ms = 255
        self.cv_state = CVState.NO_TARGET.value

        frame = self.frames.latest()
        if frame is not None and frame.seq != self._last_seq:
            self._last_seq = frame.seq
            self.ctx.frame = frame.data
            self.ctx.frame_ts = frame.timestamp
            self._process_frame()
        else:
            # No new frame: coast on the locked target (or lose it on timeout).
            self.ctx.new_observation = False
            self._coast_or_lose()

        if self.ctx.new_observation:
            self._obs_count += 1
        else:
            self._no_obs_count += 1

        # Run the estimator + solver every tick a target is locked (panels held
        # non-None). On coast ticks new_observation is False, so the estimator
        # predicts forward instead of fusing.
        if self.ctx.panels is not None:
            self.estimation.run()
            if self.ctx.estimate is not None:
                self.continuous_fire.run()
            self._publish_solution()

        self._pace(start)

    def _process_frame(self) -> None:
        """Vision + turret-frame transform for a fresh frame."""
        self.detection.run()
        panels = self.ctx.panels
        if panels is not None:
            # Drop detections with a non-finite pose. A single NaN tvec/rvec from
            # a degenerate PnP otherwise poisons the KFs permanently.
            finite = [
                p for p in panels
                if p.position is not None
                and np.all(np.isfinite(p.position))
                and (p.orientation is None or np.all(np.isfinite(p.orientation)))
            ]
            if len(finite) != len(panels):
                self.log.warning(
                    "dropped %d panel(s) with non-finite pose", len(panels) - len(finite)
                )
            panels = finite

        # At most n_panels-1 of the identical panels can face us at once; more
        # than that is a detector ghost, so drop the frame.
        has_panels = panels is not None and 0 < len(panels) < self._n_panels
        if not has_panels:
            self.ctx.new_observation = False
            self._coast_or_lose()
            return

        # Transform panel poses into the turret frame via the MCU.
        frame_ts = self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
        measured_delay_ms = int((time.perf_counter() - frame_ts) * 1000)
        # Tunable bias trim, then clamp to the u8 wire range (frameDelay_ms is a
        # single unsigned byte; >255 wraps mod 256).
        if measured_delay_ms > 255:
            self.log.warning(
                "frame_delay_ms=%d exceeds the 255ms wire limit; clamping (pipeline lagging?)",
                measured_delay_ms,
            )
        frame_delay_ms = int(
            np.clip(measured_delay_ms + config.ballistic.frame_delay_offset_ms, 0, 255)
        )
        transformation_data = self.mcu.get_transformation(frame_delay_ms)
        if transformation_data is None:
            self.log.warning("no transformation data from embedded; predicting only")
            self.ctx.new_observation = False
            self._coast_or_lose()
            return

        _turret_yaw, _turret_pitch, camera_to_turret_matrix = transformation_data
        transform_panels_to_turret_frame(panels, camera_to_turret_matrix)
        self._locked_panels = panels
        self.ctx.panels = panels
        self.ctx.new_observation = True
        self.target_timeout.reset()
        self._plot_transformed_x(panels)

    def _coast_or_lose(self) -> None:
        """No usable observation this tick: coast on the last panels, or lose on timeout."""
        if self.target_timeout.is_expired:
            self._on_target_lost()
        else:
            # Hold the last turret-frame panels so the estimator keeps predicting
            # (None if we never acquired a target).
            self.ctx.panels = self._locked_panels

    def _on_target_lost(self) -> None:
        """Timeout with no observation: drop the lock and reset the estimator."""
        if self._locked_panels is not None:
            self.log.info("target lost (%.0fms timeout) -> estimator reset",
                          self.target_timeout.duration * 1e3)
        self._locked_panels = None
        self.ctx.panels = None
        self.estimation.reset()

    def _publish_solution(self) -> None:
        """Stage ctx.solution for the MCU and pick the wire cv_state.

        A target is locked here. With a confident solution we forward pitch/yaw
        and report CONTINUOUS_FIRE (slow spin -> fire the whole time). With no
        solution, or an out-of-range one, we report AIMING (aim, hold fire).
        NO_TARGET is reserved for "no target at all" (handled by execute()).
        """
        solution = self.ctx.solution
        if solution is None:
            self.cv_state = CVState.AIMING.value
            return
        self._last_pitch = solution.pitch
        self._last_yaw = solution.yaw + np.deg2rad(config.ballistic.yaw_offset)
        self.alignment_time_ms = solution.alignment_time_ms
        self.cv_state = (
            CVState.CONTINUOUS_FIRE.value if solution.is_confident else CVState.AIMING.value
        )

    def _plot_transformed_x(self, panels: List[ArmorPanel]) -> None:
        """Push the closest panel's turret-frame x (cm) to the live debug plot."""
        if self._queue is None or not panels:
            return
        closest = min(panels, key=lambda p: float(np.linalg.norm(p.position)))
        try:
            self._queue.put_nowait(float(closest.position[0]))
        except Exception:  # full queue: drop the sample, never block the loop
            pass

    def _pace(self, start: float) -> None:
        """Sleep only enough to keep the loop under the ``loop_hz`` cap."""
        remaining = self._period - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    def update(self) -> None:
        """Send the latest solution to the MCU and service the display."""
        self.log.debug(
            "to embedded: pitch=%.2fdeg yaw=%.2fdeg align=%dms cv_state=%d",
            np.rad2deg(self._last_pitch),
            np.rad2deg(self._last_yaw),
            self.alignment_time_ms,
            self.cv_state,
        )
        self.mcu.send_solution(
            pitch=self._last_pitch,
            yaw=self._last_yaw,
            time_until_fire=self.alignment_time_ms,
            cv_state=self.cv_state,
        )
        self._log_fps()
        if config.log.display_live_frames:
            display.show_windows()

    def _log_fps(self) -> None:
        """Log observation vs no-observation FPS once per ~1s window."""
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed < 1.0:
            return
        self.log.info(
            "fps: obs=%.0f (new observations), no_obs=%.0f (predict/re-send)",
            self._obs_count / elapsed,
            self._no_obs_count / elapsed,
        )
        self._obs_count = 0
        self._no_obs_count = 0
        self._fps_window_start = time.perf_counter()


if __name__ == "__main__":
    from src.core.orchestrator import launch_system

    processes = launch_system([SpinningTargetAutoAimEngine])
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
