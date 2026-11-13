"""EKF auto-aim engine: one 10-state full-robot EKF instead of the KF state machine.

Selected with ``config.autoaim.engine: ekf`` (``src/core/orchestrator.py``).

Per fresh frame:
    detect (LightPanelDetectorModule: light-bar pixel corners, NO PnP)
    -> classify -> sticky target (closeness = light-bar pixel length)
    -> MCU camera->turret transform at the frame's capture time
    -> engine-level filter lifecycle (learner, 2026-11-12: wide logic lives here,
       not in the modules):
         * no accepted update for LOST_TIME_S -> reset the EKF;
         * EKF uninitialised -> PnP (IPPE) on the target's panels, seed
           (two adjacent panels -> seed_pair, else one panel); the seed frame
           is not also an update;
         * target lost (no detection for target_reset_timeout_ms) / switched -> reset
    -> EKF predict to frame_ts + update on the target's panels (pixel corners,
       extrinsic inside h(x))
Every tick with a target:
    |omega| > omega_spin_threshold -> FullStateShotTimingModule on the remapped
                                      state (CVState.SHOT_TIMING)
    otherwise                      -> EkfContinuousFireModule, the ported
                                      Full-Robot-EKF solver (CONTINUOUS_FIRE)

Dropped vs ``full_state_autoaim.py`` (learner decision): radii/height-delta
estimators, PanelTracking, parity anchoring, and the PARAMETER_ESTIMATION /
FULL_STATE_INIT state machine -- the EKF estimates ra, rb, h directly.

Shoot gate (PLACEHOLDER, not yet decided with the learner): fire only when the
solution is confident and the EKF has passed its own tracking threshold
(``FullRobotEKF.tracking_thres`` accepted updates since the last seed).
"""

import time
from multiprocessing import Queue
from typing import List, Optional

import numpy as np

from src.core.engine import Engine
from src.drivers.mcu import McuDriver
from src.drivers.video_stream import CameraDriver
from src.subsystems.ballistics.ekf_continuous_fire import EkfContinuousFireModule
from src.subsystems.ballistics.full_state_shot_timing import FullStateShotTimingModule
from src.subsystems.classification import RobotClassificationModule
from src.subsystems.display import display
from src.subsystems.embedded_communicator import CVState
from src.subsystems.estimation.ekf import model, pnp_seed
from src.subsystems.estimation.ekf.draw import draw_ekf_overlay
from src.subsystems.estimation.ekf.model import (
    LOG_RA,
    LOG_RB,
    OMEGA,
    THETA,
    VX,
    VY,
    ZA,
    H,
    X,
    Y,
)
from src.subsystems.estimation.ekf.module import EkfEstimationModule
from src.subsystems.targeting import TargetingModule, _closest_panel_distance
from src.subsystems.vision.classical_detector.light_module import (
    LightPanelDetectorModule,
)
from src.toolbox.globals import config
from src.toolbox.timeout import Timeout
from src.types.autoaim import ArmorPanel, EkfAutoAimContext, EnemyRobot

# Reset + re-seed when the EKF has accepted no update for this long while the
# target is still detected (awakening's lost_time_thres; learner, 2026-11-11).
LOST_TIME_S = 0.5


class EkfAutoAimEngine(Engine[EkfAutoAimContext]):
    """Auto-aim with the full-robot EKF (turret-frame state, pixel-corner measurements)."""

    drivers = {"frames": CameraDriver, "mcu": McuDriver}

    def __init__(
        self, driver_registry: Optional[dict] = None, queue: Optional[Queue] = None
    ) -> None:
        self._queue = queue  # accepted for orchestrator parity; no live plot yet
        self.ctx = EkfAutoAimContext()

        self.detection = LightPanelDetectorModule(self.ctx)
        self.classification = RobotClassificationModule(self.ctx)
        self.targeting = TargetingModule(self.ctx)
        self.estimation = EkfEstimationModule(self.ctx)
        self.continuous_fire = EkfContinuousFireModule(self.ctx)
        self.shot_timing = FullStateShotTimingModule(self.ctx)

        super().__init__(
            modules=[
                self.detection,
                self.classification,
                self.targeting,
                self.estimation,
                self.continuous_fire,
                self.shot_timing,
            ],
            context_type=EkfAutoAimContext,
            driver_registry=driver_registry,
        )

    # ------------------------------------------------------------------

    def initialize(self) -> None:
        cfg = config.autoaim
        self.target_timeout = Timeout(duration=cfg.target_reset_timeout_ms / 1e3)
        self._switch_ratio = float(cfg.target_switch_distance_ratio)
        self._period = 1.0 / float(cfg.loop_hz)
        self._omega_spin = float(config.ballistic.omega_spin_threshold)
        self._yaw_offset = float(np.deg2rad(config.ballistic.yaw_offset))

        self.mcu = self.driver("mcu")
        self.frames = self.driver("frames")
        self._last_seq = -1
        self._init_state()

    def _init_state(self) -> None:
        """Per-run bookkeeping (split out so tests can drive a non-started engine)."""
        self._last_pitch = float(np.deg2rad(-10))
        self._last_yaw = 0.0
        self.alignment_time_ms = 255
        self.cv_state = CVState.NO_TARGET.value
        self._current_target_name: Optional[str] = None
        self._prev_target_robot: Optional[EnemyRobot] = None
        self._obs_count = 0
        self._no_obs_count = 0
        self._fps_window_start = time.perf_counter()
        self.n_ekf_resets = 0
        self.seed_mode: Optional[str] = None

    # ------------------------------------------------------------------
    # Target lock
    # ------------------------------------------------------------------

    def _on_target_lost(self) -> None:
        self.ctx.target_robot = None
        self._prev_target_robot = None
        if self._current_target_name is not None:
            self.log.info("lost %r; EKF reset", self._current_target_name)
        self._current_target_name = None
        self.estimation.reset()

    def _on_target_switch(self, name: str) -> None:
        self.log.info("target switch %r -> %r; EKF reset", self._current_target_name, name)
        self._current_target_name = name
        self.estimation.reset()

    def _robot_by_name(self, name: Optional[str]) -> Optional[EnemyRobot]:
        if name is None:
            return None
        return {"sentry": self.ctx.sentry, "hero": self.ctx.hero, "standard": self.ctx.standard}.get(name)

    def _select_target(self) -> bool:
        """Sticky target selection (same policy as the full-state engine).

        Returns True iff the chosen target was freshly observed this frame.
        """
        self.targeting.run()
        closest = self.ctx.target_robot
        locked_name = self._current_target_name
        locked = self._robot_by_name(locked_name)

        if locked is not None and locked.panels:
            self.target_timeout.reset()
            target: Optional[EnemyRobot] = locked
            if (
                closest is not None
                and closest.name != locked_name
                and _closest_panel_distance(closest)
                < self._switch_ratio * _closest_panel_distance(locked)
            ):
                target = closest
            fresh = True
        elif self.target_timeout.is_expired or locked_name is None:
            target = closest
            fresh = closest is not None
            if fresh:
                self.target_timeout.reset()
        else:
            target = self._prev_target_robot
            fresh = False

        self.ctx.target_robot = target
        if target is None:
            if self.target_timeout.is_expired:
                self._on_target_lost()
            return False
        if target.name != self._current_target_name:
            self._on_target_switch(target.name)
        self._prev_target_robot = target
        return fresh

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self) -> None:
        start = time.perf_counter()
        self.ctx.start_loop_time = start
        self.ctx.solution = None
        self.alignment_time_ms = 255
        self.cv_state = CVState.NO_TARGET.value

        frame = self.frames.latest()
        fresh_frame = frame is not None and frame.seq != self._last_seq
        if fresh_frame:
            self._last_seq = frame.seq
            self.ctx.frame = frame.data
            self.ctx.frame_ts = frame.timestamp
            self._process_frame()
        else:
            self.ctx.new_observation = False
            if self.target_timeout.is_expired:
                self._on_target_lost()

        if self.ctx.new_observation:
            self._obs_count += 1
        else:
            self._no_obs_count += 1

        if self.ctx.target_robot is not None:
            self._publish_solution(self._run_estimation_and_ballistics())
        if fresh_frame and config.log.display_live_frames:
            self._draw_overlay()

        self._pace(start)

    def _run_estimation_and_ballistics(self) -> int:
        """EKF step, then shot timing or continuous fire by |omega|. Returns the fire mode."""
        self.estimation.run()
        if self.ctx.ekf_estimate is None:
            return CVState.AIMING.value  # target present, EKF not seeded yet
        omega = abs(float(self.ctx.ekf_estimate.x[OMEGA]))
        if omega > self._omega_spin:
            self.shot_timing.run()
            return CVState.SHOT_TIMING.value
        self.continuous_fire.current_yaw = self._last_yaw - self._yaw_offset
        self.continuous_fire.run()
        return CVState.CONTINUOUS_FIRE.value

    def _process_frame(self) -> None:
        """Vision + sticky targeting + MCU transform for a fresh frame."""
        self.detection.run()
        panels = self.ctx.panels
        if panels is not None:
            panels = [
                p for p in panels
                if p.light_corners is not None and np.all(np.isfinite(p.light_corners))
            ]
            self.ctx.panels = panels
        if not panels or len(panels) >= 3:  # >2 panels of one robot can't face us
            self.ctx.new_observation = False
            if self.target_timeout.is_expired:
                self._on_target_lost()
            elif self._prev_target_robot is not None:
                self.ctx.target_robot = self._prev_target_robot
            return

        self.classification.run()
        if not self._select_target() or self.ctx.target_robot is None:
            self.ctx.new_observation = False
            return

        frame_ts = self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
        measured_delay_ms = int((time.perf_counter() - frame_ts) * 1000)
        if measured_delay_ms > 255:
            self.log.warning(
                "frame_delay_ms=%d exceeds the 255ms wire limit; clamping (pipeline lagging?)",
                measured_delay_ms,
            )
        frame_delay_ms = int(
            np.clip(measured_delay_ms + config.ballistic.frame_delay_offset_ms, 0, 255)
        )
        transformation = self.mcu.get_transformation(frame_delay_ms)
        if transformation is None:
            self.log.warning("no transformation data from embedded; predicting only")
            self.ctx.new_observation = False
            return
        _yaw, _pitch, camera_to_turret = transformation
        self.ctx.camera_to_turret = np.asarray(camera_to_turret, dtype=np.float64)
        self.ctx.new_observation = True
        self._maintain_filter(self.ctx.target_robot.panels, frame_ts, self.ctx.camera_to_turret)

    def _maintain_filter(
        self, panels: List[ArmorPanel], frame_ts: float, cam_to_turret: np.ndarray
    ) -> None:
        """Lost reset + PnP seed, before the estimation module's predict/update."""
        est = self.estimation
        if (est.ekf.initialized and est.last_accept_ts is not None
                and frame_ts - est.last_accept_ts > LOST_TIME_S):
            self.log.warning("EKF: no accepted update for %.2f s; reset + re-seed",
                             frame_ts - est.last_accept_ts)
            est.reset()
            self.n_ekf_resets += 1
        if est.ekf.initialized:
            return
        K, dist = est.intrinsics
        cands = [pnp_seed.panel_candidates(p.light_corners, K, dist, cam_to_turret)
                 for p in panels[:2]]          # closest first (classification sort)
        seed = pnp_seed.choose_seed(cands)
        if seed is None:
            self.log.debug("EKF seed: PnP found no camera-facing pose")
            return
        est.seed(seed, frame_ts)
        self.seed_mode = seed.mode
        self.log.info("EKF seeded (%s)", seed.mode)

    def _draw_overlay(self) -> None:
        """Full-Robot-EKF-style debug view on the detector's frame (once per new frame)."""
        est = self.estimation
        ekf = est.ekf
        res = est.last_result if self.ctx.new_observation else None
        x = ekf.x if ekf.initialized else None
        cam = None
        if x is not None and self.ctx.camera_to_turret is not None:
            K, dist = est.intrinsics
            cam = model.Camera(K, dist, self.ctx.camera_to_turret)
        dets = [p.light_corners for p in (self.ctx.panels or []) if p.light_corners is not None]
        target = self.ctx.target_robot.name if self.ctx.target_robot is not None else "-"
        hud = [
            f"target: {target}   det: {len(dets)}   mode: {CVState(self.cv_state).name}",
            f"init:   {self.seed_mode if x is not None else 'waiting'}   resets {self.n_ekf_resets}",
            f"upd:    {'yes' if res is not None and res.accepted else 'no'}"
            f"   n {ekf.accepted_count}   EKF {est.last_filter_ms:5.2f} ms",
        ]
        if res is not None and res.innovation is not None:
            hud.append(f"err:    {float(np.sqrt(np.mean(res.innovation ** 2))):6.2f} px")
        if x is not None:
            hud += [
                f"x:      {x[X]:7.3f} m",
                f"y:      {x[Y]:7.3f} m",
                f"z_a:    {x[ZA]:7.3f} m",
                f"|v|:    {float(np.hypot(x[VX], x[VY])):6.3f} m/s",
                f"omega:  {x[OMEGA]:7.3f} rad/s",
                f"theta:  {np.degrees(x[THETA]):7.2f} deg",
                f"ra:     {np.exp(x[LOG_RA]):6.3f} m",
                f"rb:     {np.exp(x[LOG_RB]):6.3f} m",
                f"h:      {x[H]:6.3f} m",
                f"vis:    {model.visible_panels(x, cam) if cam is not None else '-'}",
            ]
        draw_ekf_overlay(display.windows["main"].img, x, cam, dets, hud)

    def _publish_solution(self, fire_mode: int) -> None:
        solution = self.ctx.solution
        if solution is None:
            self.cv_state = CVState.AIMING.value
            return
        self._last_pitch = solution.pitch
        self._last_yaw = solution.yaw + self._yaw_offset
        self.alignment_time_ms = solution.alignment_time_ms
        self.cv_state = self._shoot_decision(fire_mode, solution.is_confident)

    def _shoot_decision(self, fire_mode: int, is_confident: bool) -> int:
        """PLACEHOLDER gate: confident solution + EKF past its tracking threshold."""
        if not is_confident:
            return CVState.AIMING.value
        est = self.ctx.ekf_estimate
        if est is None or est.accepted_updates < self.estimation.ekf.tracking_thres:
            return CVState.AIMING.value
        return fire_mode

    def _pace(self, start: float) -> None:
        remaining = self._period - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    def update(self) -> None:
        est = self.ctx.ekf_estimate
        if est is not None and self.log.isEnabledFor(10):  # DEBUG
            x = est.x
            self.log.debug(
                "ekf: c=(%.2f,%.2f,%.2f)m th=%.0fdeg w=%.2frad/s ra=%.3f rb=%.3f h=%.3f n=%d",
                x[X], x[Y], x[ZA], np.degrees(x[THETA]), x[OMEGA],
                np.exp(x[LOG_RA]), np.exp(x[LOG_RB]), x[H], est.accepted_updates,
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
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed < 1.0:
            return
        self.log.info(
            "fps: obs=%.0f (new target observations), no_obs=%.0f (predict/re-send)",
            self._obs_count / elapsed,
            self._no_obs_count / elapsed,
        )
        self._obs_count = 0
        self._no_obs_count = 0
        self._fps_window_start = time.perf_counter()
        self.n_ekf_resets = 0
        self.seed_mode: Optional[str] = None
