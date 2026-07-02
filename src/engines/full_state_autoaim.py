"""Full-state auto-aim engine with online per-robot constant learning.

Replaces the hardcoded panel-orbit radius and config z_offset of the archived
engine (``archive_full_state_autoaim.py``) with constants *learned online* per
enemy robot and persisted across runs. Kalman-filter-only (the GPU particle
filter lives only in the archived engine).

STATE MACHINE (one state per enemy-robot name; every transition lives in this
file -- modules know nothing about states):

    PARAMETER_ESTIMATION  (no saved constants for this robot)
        Runs the full-state KF with the configured initial radius guess so
        PanelTrackingModule can assign panel ids; learns the per-parity orbit
        radii (RadiiEstimatorModule) and the signed inter-pair height delta
        (PanelHeightDeltaModule). Meanwhile AIMS AT THE CLOSEST PANEL directly
        (SinglePanelEstimation + SinglePanelBallistics -- actual panel z, no
        robot geometry needed).
        T1 -> FULL_STATE_INIT: both learning KFs converged (variance below
            config thresholds, enough updates) => canonicalize + save to the
            JsonStore.
        T5 (self-loop) target lost: full-state/single-panel KFs + tracker
            reset, but the radii/height KFs are KEPT (static geometry). Panel
            ids restart with the new theta track, so on the next two-panel
            measurement ``match_parity`` checks whether the kept state is
            parity-swapped relative to the new ids and, if so, flips the
            tracker's parity offset (the single place parity flips).

    FULL_STATE_INIT  (constants exist; panel-id parity not yet anchored)
        Single-panel aiming only (ids are track-relative -- "panel 0" from a
        previous session/track is meaningless, see ``robot_constants``).
        T2 -> FULL_STATE_TRACKING: two ~90deg-apart panels visible =>
            ``anchor_parity`` maps the saved r_low/r_high onto live parity by
            the RELATIVE z of the two panels (platforms don't break it;
            degenerate height falls back to a one-shot radii measurement),
            then the full-state KF is re-seeded from the anchor panel with the
            correct per-parity radii + aim geometry.
        T4 (self-loop) target lost: single-panel KF reset.

    FULL_STATE_TRACKING  (anchored)
        PanelTracking maintains ids each frame (so single-panel frames still
        pick the right back-projection radius); full-state KF estimates; the
        ballistic module is chosen by |omega| vs omega_spin_threshold
        (shot timing vs continuous fire), aiming at ``estimate.aim_z`` -- the
        live mid-height between the panel pairs.
        T3 -> FULL_STATE_INIT: target lost (ids are track-relative => must
            re-anchor).

    T6 (any state): the *target robot name* changes => per-robot dicts swap in
    the new robot's state; track-bound filters reset; a TRACKING robot demotes
    to FULL_STATE_INIT.

OWNERSHIP: this engine owns the state machine, all disk I/O for constants
(JsonStore), which module.run() calls happen each iteration, MCU I/O, frame
pacing, and the target-lost timeout. Modules receive constants via explicit
setters at transition time -- they never fetch config/disk/state themselves.

Solution handling / shoot decision: when a target exists the engine forwards
pitch/yaw and picks the wire cv_state in ``_publish_solution``. It reports a
firing state (SHOT_TIMING / CONTINUOUS_FIRE) only when the ballistic solution is
confident (in range, real arc) AND the estimator's predicted lead-horizon gate
radius is below ``ballistic.shoot_gate_radius_threshold``; otherwise it reports
``CVState.AIMING`` (aim, hold fire). NO_TARGET is reserved for "no target at
all". The gate radius is the 1-sigma 2-D position spread of the F·P·Fᵀ+Q
prediction (see ``PositionKF.gate_radius``); it is stored on the engine as
``gate_radius_cm`` each tick.
"""

import logging
import time
from enum import Enum, auto
from multiprocessing import Queue
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from src.core.engine import Engine
from src.drivers.mcu import McuDriver
from src.drivers.video_stream import CameraDriver
from src.subsystems.ballistics.full_state_continuous_fire import (
    FullStateContinuousFireModule,
)
from src.subsystems.ballistics.full_state_shot_timing import FullStateShotTimingModule
from src.subsystems.ballistics.single_panel import SinglePanelBallisticModule
from src.subsystems.classification import RobotClassificationModule
from src.subsystems.display import display
from src.subsystems.embedded_communicator import CVState
from src.subsystems.estimation.full_state.kf import (
    KalmanFilterEstimationModule,
    _default_full_state_kf,
)
from src.subsystems.estimation.height_delta import PanelHeightDeltaModule
from src.subsystems.estimation.radii import RadiiEstimatorModule
from src.subsystems.estimation.robot_constants import (
    RobotConstants,
    anchor_parity,
    canonicalize,
    load_constants,
    match_parity,
    measure_pair_by_parity,
    save_constants,
)
from src.subsystems.estimation.single_panel import SinglePanelEstimationModule
from src.subsystems.panel_tracking import PanelTrackingModule
from src.subsystems.targeting import TargetingModule, _closest_panel_distance
from src.subsystems.vision import ClassicalDetectorModule
from src.toolbox.globals import absolute_path_to, config
from src.toolbox.storage import JsonStore
from src.toolbox.timeout import Timeout
from src.types.autoaim import EnemyRobot, FullStateAutoAimContext

METERS_TO_CM = 100
_ADJACENT_YAW_TOL = np.radians(20.0)  # two-panel gate: |yaw separation - 90deg|


class AimState(Enum):
    """Per-robot aim state (see the module docstring for the transitions)."""

    PARAMETER_ESTIMATION = auto()  # state 1: learning radii + height delta
    FULL_STATE_INIT = auto()       # state 2a: constants loaded, parity unanchored
    FULL_STATE_TRACKING = auto()   # state 2b: anchored, full-state aiming


class FullStateAutoAimEngine(Engine[FullStateAutoAimContext]):
    """Auto-aim engine: learns each robot's geometry, then full-state aims with it."""

    drivers = {"frames": CameraDriver, "mcu": McuDriver}

    def __init__(
        self, driver_registry: Optional[dict] = None, queue: Optional[Queue] = None
    ) -> None:
        self._queue = queue
        self.ctx = FullStateAutoAimContext()

        est_cfg = config.estimation
        self.detection = ClassicalDetectorModule(self.ctx)
        self.classification = RobotClassificationModule(self.ctx)
        self.targeting = TargetingModule(self.ctx)
        self.panel_tracking = PanelTrackingModule(self.ctx)
        self.estimation = KalmanFilterEstimationModule(self.ctx)
        self.radii_estimator = RadiiEstimatorModule(
            self.ctx,
            initial_guess=est_cfg.initial_radius_guess,
            gate_confidence=est_cfg.radii_gate_confidence,
        )
        self.height_delta_estimator = PanelHeightDeltaModule(self.ctx)
        self.single_panel_estimation = SinglePanelEstimationModule(
            self.ctx, jump_reinit_cm=est_cfg.panel_jump_reinit_cm
        )
        self.single_panel_ballistics = SinglePanelBallisticModule(self.ctx)
        self.shot_timing = FullStateShotTimingModule(self.ctx)
        self.continuous_fire = FullStateContinuousFireModule(self.ctx)

        super().__init__(
            modules=[
                self.detection,
                self.classification,
                self.targeting,
                self.panel_tracking,
                self.estimation,
                self.radii_estimator,
                self.height_delta_estimator,
                self.single_panel_estimation,
                self.single_panel_ballistics,
                self.shot_timing,
                self.continuous_fire,
            ],
            context_type=FullStateAutoAimContext,
            driver_registry=driver_registry,
        )

    # ------------------------------------------------------------------
    # Initialization (child process)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create child-process resources: estimators, constants store, MCU, frames."""
        self._init_estimators()
        self._init_state_machine(JsonStore(absolute_path_to.robot_constants))

        cfg = config.autoaim
        self.target_timeout = Timeout(duration=cfg.target_reset_timeout_ms / 1e3)
        self._switch_ratio = float(cfg.target_switch_distance_ratio)
        # Loop runs as fast as it can with this period as a CAP (we only sleep
        # to avoid exceeding loop_hz; detection ticks run longer).
        self._period = 1.0 / float(cfg.loop_hz)

        # MCU access goes through the shared McuDriver (serial/ros/mock chosen
        # by the MCU= profile); the engine never opens the UART itself.
        self.mcu = self.driver("mcu")
        # Fallback pitch/yaw sent when no target is available.
        self._last_pitch: float = float(np.deg2rad(-10))
        self._last_yaw: float = 0.0
        self.alignment_time_ms: int = 255
        self.cv_state: int = CVState.NO_TARGET.value
        # Shoot-confidence gate: the active estimator's predicted 1-sigma position
        # gate radius (cm) at the lead horizon, refreshed each tick by the active
        # state pipeline and consumed in _publish_solution. inf = not yet computed
        # / no track -> hold fire.
        self._lead: float = float(config.ballistic.lookahead_time)
        self._shoot_gate_threshold: float = float(config.ballistic.shoot_gate_radius_threshold)
        self.gate_radius_cm: float = float("inf")

        # Frames come from the shared CameraDriver via this FrameReader handle.
        self.frames = self.driver("frames")
        self._last_seq = -1

        # Two separate FPS tallies logged at INFO: ticks with a fresh target
        # observation vs predict/re-send ticks with none. (_prev_target_robot,
        # the robot we coast on within the lock timeout, is reset in
        # _init_state_machine.)
        self._obs_count = 0
        self._no_obs_count = 0
        self._fps_window_start = time.perf_counter()

    def _init_estimators(self) -> None:
        """Build the full-state KF and inject it + the initial radii guess."""
        guess = float(config.estimation.initial_radius_guess)
        estimator = _default_full_state_kf(r=guess)
        self.estimation.set_estimator(estimator)
        self.estimation.set_panel_radii(guess, guess)

    def _init_state_machine(self, store: JsonStore) -> None:
        """Load saved constants and reset all per-robot state-machine bookkeeping.

        Split out from initialize() so unit tests can drive the state machine
        on a non-started engine with a tmp-path store and no hardware.
        """
        self._store = store
        if config.estimation.force_reestimation:
            self.log.info("force_reestimation: ignoring saved robot constants")
            self._constants: Dict[str, RobotConstants] = {}
        else:
            self._constants = load_constants(store)
            if self._constants:
                self.log.info("loaded robot constants: %s", sorted(self._constants))
        self._state: Dict[str, AimState] = {}
        self._needs_parity_anchor: Dict[str, bool] = {}
        self._current_target_name: Optional[str] = None
        self._prev_target_robot: Optional[EnemyRobot] = None

    # ------------------------------------------------------------------
    # State machine (ALL transition logic lives in this section)
    # ------------------------------------------------------------------

    def _state_for(self, name: str) -> AimState:
        """Current state for a robot, defaulting by whether constants exist."""
        if name not in self._state:
            self._state[name] = (
                AimState.FULL_STATE_INIT
                if name in self._constants
                else AimState.PARAMETER_ESTIMATION
            )
            self.log.info("robot %r enters %s", name, self._state[name].name)
        return self._state[name]

    def _on_target_lost(self) -> None:
        """T3/T4/T5: the target timed out -- reset everything bound to the theta track."""
        self.ctx.target_robot = None
        self._prev_target_robot = None
        name = self._current_target_name
        self._current_target_name = None
        if name is None:
            return

        # Track-bound state is broken in every aim state.
        self.estimation.reset()
        self.single_panel_estimation.reset()
        self.panel_tracking.reset()

        state = self._state_for(name)
        if state == AimState.FULL_STATE_TRACKING:
            # T3: ids are track-relative -> parity must be re-anchored.
            self.estimation.clear_aim_geometry()
            self._state[name] = AimState.FULL_STATE_INIT
            self.log.info("T3: lost %r while TRACKING -> FULL_STATE_INIT", name)
        elif state == AimState.PARAMETER_ESTIMATION:
            # T5: keep the radii/height KFs (static geometry), but their parity
            # keying may not match the next theta track's ids.
            self._needs_parity_anchor[name] = True
            self.log.info("T5: lost %r in PARAMETER_ESTIMATION; will re-anchor parity", name)

    def _on_target_switch(self, name: str) -> None:
        """T6: the targeted robot's name changed -- swap in its per-robot state."""
        previous = self._current_target_name
        self._current_target_name = name

        # Same track-bound resets as a loss (the theta track belonged to the
        # previous robot).
        self.estimation.reset()
        self.single_panel_estimation.reset()
        self.panel_tracking.reset()
        self.estimation.clear_aim_geometry()

        if previous is not None and self._state_for(previous) == AimState.FULL_STATE_TRACKING:
            self._state[previous] = AimState.FULL_STATE_INIT
        if self._state_for(name) == AimState.PARAMETER_ESTIMATION:
            self._needs_parity_anchor[name] = True
        self.log.info("T6: target switch %r -> %r (%s)", previous, name, self._state_for(name).name)

    def _robot_by_name(self, name: Optional[str]) -> Optional[EnemyRobot]:
        """The freshly-classified robot for ``name`` this frame (None if absent)."""
        if name is None:
            return None
        return {
            "sentry": self.ctx.sentry,
            "hero": self.ctx.hero,
            "standard": self.ctx.standard,
        }.get(name)

    def _select_target(self) -> bool:
        """Sticky target selection on a fresh frame (runs after classification).

        Keeps the locked robot and only switches when (a) a *different* robot is
        much closer than the lock, or (b) the lock has timed out. While the lock
        is alive but its panels are missing this frame, coasts on the remembered
        past robot. Returns True iff the resulting target was freshly observed
        this frame (=> a real observation); False means we are coasting and the
        estimator should predict-only.
        """
        self.targeting.run()  # ctx.target_robot = closest robot WITH panels, or None
        closest = self.ctx.target_robot

        locked_name = self._current_target_name
        locked = self._robot_by_name(locked_name)
        locked_seen = locked is not None and bool(locked.panels)

        if locked_seen:
            assert locked is not None  # implied by locked_seen; narrows for mypy
            self.target_timeout.reset()
            target: Optional[EnemyRobot] = locked
            if (
                closest is not None
                and closest.name != locked_name
                and _closest_panel_distance(closest)
                < self._switch_ratio * _closest_panel_distance(locked)
            ):
                target = closest  # a different robot is dramatically closer
            fresh = True
        elif self.target_timeout.is_expired or locked_name is None:
            target = closest  # lock lost / none yet: take the closest (may be None)
            fresh = closest is not None
            if fresh:
                self.target_timeout.reset()
        else:
            target = self._prev_target_robot  # coast on the remembered robot
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

    def _maybe_reanchor_parity(self, name: str) -> None:
        """T5 follow-up: re-key the kept radii/height state onto the new id parity.

        Runs once after a state-1 target loss, on the first fresh two-panel
        measurement: compares it against the kept KF means under identity vs
        swap; a better swap fit flips the tracker's parity offset (which also
        re-labels the current frame's ids before the learning modules run).
        """
        if not self._needs_parity_anchor.get(name):
            return
        radii = self.ctx.radii_estimate
        height = self.ctx.height_delta_estimate
        if radii is None or height is None or self.ctx.target_robot is None:
            self._needs_parity_anchor[name] = False  # nothing kept: ids are fresh truth
            return
        measurement = measure_pair_by_parity(self.ctx.target_robot.panels)
        if measurement is None:
            return  # keep waiting for an adjacent pair
        stored = np.array([radii.r_even, radii.r_odd, height.dz])
        measured = np.array([measurement.r_a, measurement.r_b, measurement.dz])
        if match_parity(stored, measured):
            self.panel_tracking.set_parity_offset(1)
            self.panel_tracking.run()  # re-label this frame's ids with the flip
            self.log.info("T5: parity swap detected for %r; tracker offset flipped", name)
        self._needs_parity_anchor[name] = False

    def _check_constants_converged(self, name: str) -> None:
        """T1: PARAMETER_ESTIMATION -> FULL_STATE_INIT when both KFs converged."""
        radii = self.ctx.radii_estimate
        height = self.ctx.height_delta_estimate
        if radii is None or height is None:
            return
        cfg = config.estimation
        converged = (
            radii.n_updates >= cfg.min_constant_updates
            and height.n_updates >= cfg.min_constant_updates
            and radii.var_even < cfg.radii_var_threshold
            and radii.var_odd < cfg.radii_var_threshold
            and height.var < cfg.height_var_threshold
        )
        if not converged:
            return

        constants = canonicalize(
            radii.r_even,
            radii.r_odd,
            height.dz,
            height_degenerate_cm=cfg.height_degenerate_cm,
            radii_degenerate_cm=cfg.radii_degenerate_cm,
        )
        self._constants[name] = constants
        save_constants(self._store, name, constants)
        self._state[name] = AimState.FULL_STATE_INIT
        self.log.info(
            "T1: constants for %r converged (r_low=%.1f r_high=%.1f dz=%.1f) -> FULL_STATE_INIT",
            name,
            constants.r_low,
            constants.r_high,
            constants.height_delta,
        )

    def _try_anchor(self, name: str) -> None:
        """T2: FULL_STATE_INIT -> FULL_STATE_TRACKING once two adjacent panels anchor parity."""
        target = self.ctx.target_robot
        if target is None or not self.ctx.new_observation:
            return
        panels = [p for p in target.panels if p.position is not None]
        if len(panels) != 2:
            return
        a, b = panels
        pos_a, pos_b = a.position, b.position
        if pos_a is None or pos_b is None:  # filtered above; keeps narrowing happy
            return
        d_yaw = float(np.arctan2(np.sin(b.yaw - a.yaw), np.cos(b.yaw - a.yaw)))
        if abs(abs(d_yaw) - np.pi / 2) > _ADJACENT_YAW_TOL:
            return  # not a clean adjacent pair (detector ghost?)

        # Seed ids relative to the about-to-be-created theta track: the anchor
        # panel a becomes id 0 (theta seeds from a.yaw), its neighbor 1 or 3.
        a.id = 0
        b.id = 1 if d_yaw > 0 else 3
        measurement = measure_pair_by_parity(panels)
        if measurement is None:
            return

        constants = self._constants[name]
        anchor = anchor_parity(
            constants,
            z_even=float(pos_a[2]),
            z_odd=float(pos_b[2]),
            measured=measurement,
        )
        self.estimation.set_panel_radii(anchor.r_even, anchor.r_odd)
        self.estimation.set_aim_geometry(constants.height_delta, anchor.low_parity)
        self.estimation.reinit_from_panel(a)
        self.panel_tracking.reset()  # offset 0: ids are now defined by this track
        self._state[name] = AimState.FULL_STATE_TRACKING
        self.log.info(
            "T2: anchored %r (low_parity=%d r_even=%.1f r_odd=%.1f) -> FULL_STATE_TRACKING",
            name,
            anchor.low_parity,
            anchor.r_even,
            anchor.r_odd,
        )

    # ------------------------------------------------------------------
    # Per-state pipelines (which modules run; no transition conditions here
    # beyond delegating to the _check/_try methods above)
    # ------------------------------------------------------------------

    def _run_parameter_estimation(self, name: str) -> int:
        """State 1: learn constants while aiming at the closest panel."""
        if self.ctx.new_observation:
            self.panel_tracking.run()  # ids from the PREVIOUS frame's estimate
        self.estimation.run()
        if self.ctx.new_observation:
            self._maybe_reanchor_parity(name)
            self.radii_estimator.run()
            self.height_delta_estimator.run()
        self.single_panel_estimation.run()
        self.gate_radius_cm = self.single_panel_estimation.gate_radius(self._lead)
        self.single_panel_ballistics.run()
        self._check_constants_converged(name)  # T1
        return CVState.CONTINUOUS_FIRE.value

    def _run_full_state_init(self, name: str) -> int:
        """State 2a: single-panel aiming until parity can be anchored."""
        self.single_panel_estimation.run()
        self.gate_radius_cm = self.single_panel_estimation.gate_radius(self._lead)
        self.single_panel_ballistics.run()
        self._try_anchor(name)  # T2
        return CVState.CONTINUOUS_FIRE.value

    def _run_full_state_tracking(self, _name: str) -> int:
        """State 2b: full-state estimation + spin-dependent ballistics."""
        if self.ctx.new_observation:
            self.panel_tracking.run()
        self.estimation.run()
        if self.ctx.estimate is None:
            # We still have a target, just no usable estimate this tick: aim, hold
            # fire. NO_TARGET is reserved for "no target at all" (target_robot None).
            return CVState.AIMING.value
        self.gate_radius_cm = self.estimation.estimator.gate_radius(self._lead)
        omega = abs(float(self.ctx.estimate.value[5]))
        if omega > config.ballistic.omega_spin_threshold:
            self.shot_timing.run()
            return CVState.SHOT_TIMING.value
        self.continuous_fire.run()
        return CVState.CONTINUOUS_FIRE.value

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """One loop tick: process a new frame if present, else predict; then solve + send.

        Runs as fast as it can with ``loop_hz`` as a cap. A new camera frame
        (fresh ``seq``) drives the full vision pipeline; every other tick just
        predicts the estimator forward and re-solves, so ballistic angles flow
        between frames.
        """
        start = time.perf_counter()
        self.ctx.start_loop_time = start
        self.ctx.solution = None  # never reuse a stale solution
        self.alignment_time_ms = 255
        self.cv_state = CVState.NO_TARGET.value
        self.gate_radius_cm = float("inf")  # refreshed by the active state pipeline

        frame = self.frames.latest()
        if frame is not None and frame.seq != self._last_seq:
            self._last_seq = frame.seq
            self.ctx.frame = frame.data
            self.ctx.frame_ts = frame.timestamp
            self._process_frame()
        else:
            # No new frame: coast on the locked target (or lose it on timeout).
            self.ctx.new_observation = False
            if self.target_timeout.is_expired:
                self._on_target_lost()

        # Two FPS tallies: fresh target observations vs predict/re-send ticks.
        if self.ctx.new_observation:
            self._obs_count += 1
        else:
            self._no_obs_count += 1

        # ------------------------------------------------------------------
        # Dispatch to the active state's pipeline (every tick a target exists).
        # ------------------------------------------------------------------
        if self.ctx.target_robot is not None:
            # self.log.debug(f"panels: {self.ctx.target_robot.panels}")
    
            name = self.ctx.target_robot.name
            state = self._state_for(name)
            if state == AimState.PARAMETER_ESTIMATION:
                active_cv_state = self._run_parameter_estimation(name)
            elif state == AimState.FULL_STATE_INIT:
                active_cv_state = self._run_full_state_init(name)
            else:
                active_cv_state = self._run_full_state_tracking(name)
            # self.log.debug("estimate before state pipeline: %s", self.ctx.estimate)
            self._publish_solution(active_cv_state)

        self._pace(start)

    def _process_frame(self) -> None:
        """Vision + sticky targeting + turret-frame transform for a fresh frame."""
        self.detection.run()
        panels = self.ctx.panels
        if panels is not None:
            # Drop detections with a non-finite pose. A single NaN tvec/rvec from
            # a degenerate PnP otherwise poisons the KFs permanently (NaN aim_z,
            # position, or yaw survive every Kalman update until the next reinit).
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
            self.ctx.panels = finite
        # >2 panels of one robot can't face us at once: treat as detector ghosts.
        has_panels = panels is not None and 0 < len(panels) < 3

        if not has_panels:
            # No usable panels: coast on the remembered target, or lose on timeout.
            self.ctx.new_observation = False
            if self.target_timeout.is_expired:
                self._on_target_lost()
            elif self._prev_target_robot is not None:
                self.ctx.target_robot = self._prev_target_robot
            return

        self.classification.run()
        fresh = self._select_target()
        if not fresh or self.ctx.target_robot is None:
            self.ctx.new_observation = False
            return
        # Transform panel poses into the turret frame via the MCU.
        frame_ts = self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
        measured_delay_ms = int((time.perf_counter() - frame_ts) * 1000)
        # Tunable bias trim (config.ballistic.frame_delay_offset_ms), then clamp to
        # the u8 wire range so a large/negative offset can't wrap the byte. The MCU
        # protocol carries frameDelay_ms as a single unsigned byte (>255 wraps mod
        # 256), so warn if the measured age alone already blew the limit.
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
        else:
            _turret_yaw, _turret_pitch, camera_to_turret_matrix = transformation_data
            _transform_panels_to_turret_frame(panels, camera_to_turret_matrix)
            self.ctx.new_observation = True
            self._plot_transformed_x(panels)

    def _plot_transformed_x(self, panels) -> None:
        """Push the closest panel's turret-frame x (cm) to the live debug plot.

        No-op unless a plot queue was wired in (``config.log.live_plot``). Uses
        the closest panel since that's what single-panel aiming tracks.
        """
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

    def _publish_solution(self, active_cv_state: int) -> None:
        """Stage ctx.solution for the MCU and make the engine-level shoot decision.

        We have a target here (this runs only when ``target_robot`` is set), so
        the wire never carries NO_TARGET from this path -- it's either a firing
        cv_state or ``AIMING`` (aim, hold fire). NO_TARGET is reserved for "no
        target at all" (``target_robot`` None, handled by the execute() default).

        With no fresh solution this tick we still have a target, so we report
        ``AIMING`` and coast on the last pitch/yaw. With a solution we forward its
        pitch/yaw and fire only when both hold:
          * the ballistic solution is confident (in range, real arc), and
          * the predicted lead-horizon gate radius is below the configured
            threshold (the estimate is tight enough to trust).
        Otherwise -> ``AIMING`` (firmware aims without firing).
        """
        solution = self.ctx.solution
        if solution is None:
            self.cv_state = CVState.AIMING.value  # target present, no solution: hold fire
            return

        self._last_pitch = solution.pitch
        self._last_yaw = solution.yaw + np.deg2rad(config.ballistic.yaw_offset)
        self.alignment_time_ms = solution.alignment_time_ms
        self.cv_state = self._shoot_decision(active_cv_state, solution.is_confident)

    def _shoot_decision(self, active_cv_state: int, is_confident: bool) -> int:
        """Map the active state's fire mode to a wire cv_state via the confidence gate."""
        if not is_confident:
            return CVState.AIMING.value  # out of range / no ballistic arc
        if self.gate_radius_cm > self._shoot_gate_threshold:
            if self.log.isEnabledFor(logging.DEBUG):
                self.log.debug(
                    "gate radius %.1fcm > %.1fcm threshold -> AIMING (hold fire)",
                    self.gate_radius_cm,
                    self._shoot_gate_threshold,
                )
            return CVState.AIMING.value
        return active_cv_state

    def update(self) -> None:
        """Send the latest solution to the MCU and service the display."""
        self.log.debug(
            "to embedded: pitch=%.2fdeg yaw=%.2fdeg align=%dms cv_state=%d gate=%.1fcm",
            np.rad2deg(self._last_pitch),
            np.rad2deg(self._last_yaw),
            self.alignment_time_ms,
            self.cv_state,
            self.gate_radius_cm,
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
            "fps: obs=%.0f (new target observations), no_obs=%.0f (predict/re-send)",
            self._obs_count / elapsed,
            self._no_obs_count / elapsed,
        )
        self._obs_count = 0
        self._no_obs_count = 0
        self._fps_window_start = time.perf_counter()


# ---------------------------------------------------------------------------
# Helpers (not modules -- used inline in execute)
# ---------------------------------------------------------------------------


def _transform_panels_to_turret_frame(
    panels, camera_to_turret_matrix: np.ndarray
) -> None:
    """Transform panel poses from camera frame to turret/ballistic frame.

    Modifies panels in-place. Both the position and the outward-normal yaw are
    carried through the SAME relabel + camera->turret rotation, so they end up
    in one consistent frame (canonical convention -- see src/types/autoaim.py).

    NOTE: yaw is NOT a scalar ``+ turret_yaw`` of the detector's camera-relative
    yaw. Two things make that wrong. (1) ``panel.orientation`` (rvec) is still in
    RAW OpenCV camera axes (x-right, y-down, z-fwd) -- unlike ``panel.position``,
    which pnp.py already relabels to right/fwd/up via ``[v0, v2, -v1]``. (2) The
    detector yaw is the normal's azimuth about the optical axis in the camera's
    *pitched* plane, while the ballistic frame wants it from +x in the level
    plane. So we relabel the 3-D normal exactly like the position, then push it
    through the same matrix rotation. Adding ``turret_yaw`` supplied only the
    gimbal yaw and neither correction, leaving a frame-varying yaw error that the
    back-projection radius turned into a lateral (left/right) centre error.
    """
    R_ct = camera_to_turret_matrix[:3, :3]
    for panel in panels:
        if panel.position is None or panel.orientation is None:
            continue
        # tvec is in cm -> metres for the 4x4 transform, then back to cm.
        pos_m = np.append(panel.position / METERS_TO_CM, 1.0)
        panel.position = (camera_to_turret_matrix @ pos_m)[:3] * METERS_TO_CM

        # Panel +z (3rd column of R) is the outward normal in RAW OpenCV camera
        # axes. Relabel it to right/fwd/up just like pnp.py does to tvec, then
        # apply the same camera->turret rotation, and read off its azimuth.
        n_cam = cv.Rodrigues(panel.orientation)[0][:, 2]
        n_relabeled = np.array([n_cam[0], n_cam[2], -n_cam[1]])
        n_turret = R_ct @ n_relabeled
        panel.yaw = float(np.arctan2(n_turret[1], n_turret[0]))
