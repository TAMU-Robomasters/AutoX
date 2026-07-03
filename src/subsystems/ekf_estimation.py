"""EKF estimation module for the spinning-target auto-aim pipeline.

Wraps :class:`EkfTracker` (centre position + spin phase) plus a
:class:`HeightKF` (panel-centre height) in a :class:`Module` so it slots into
the engine's clipboard-style ``Context`` plumbing. Mirrors
:class:`ParticleFilterEstimationModule` (``src/subsystems/pf.py``) but consumes
``panels`` directly -- there is no classification/targeting layer because the
rig has a single spinning target.

No-observation handling (mirrors ``full_state`` KF module): the engine keeps
``ctx.panels`` non-None while a target is locked -- including brief coast ticks
when a frame has no usable panels -- and sets it to ``None`` only on true target
loss. So ``panels is None`` here means "no target": bail and re-seed next time.
Otherwise, on ``new_observation`` we predict+update, and on a coast tick we
predict-only (``update_with_no_observation``) so the centre, spin phase, and
height all keep extrapolating between detections instead of freezing.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.ekf import EkfTracker
from src.subsystems.estimation.filters import HeightKF
from src.toolbox.globals import config
from src.types.autoaim import (
    ArmorPanel,
    EkfAutoAimContext,
    RobotStateEstimate,
)


def _default_ekf_tracker() -> EkfTracker:
    """Build an ``EkfTracker`` from ``config.spinning_target``."""
    st = config.spinning_target
    omega0 = float(st.omega_rpm) * 2.0 * np.pi / 60.0
    return EkfTracker(
        radius=float(st.radius_cm),
        omega0=omega0,
        n_panels=int(st.n_panels),
    )


def _default_height_kf() -> HeightKF:
    """Panel-centre height filter (random walk, low process noise).

    All 3 panels sit at the same height, so the measurement each frame is just
    the observed panel's turret-frame z; the KF smooths per-frame z noise and
    rides through no-observation ticks. Reuses the full-state engine's tuned
    ``estimation.height_kf_{q,r}`` values.
    """
    return HeightKF(
        q=float(config.estimation.height_kf_q),
        r=float(config.estimation.height_kf_r),
    )


class EkfEstimationModule(Module[EkfAutoAimContext]):
    """Estimate spinning-target state (position + spin + height) with linear KFs."""

    def __init__(self, context: EkfAutoAimContext):
        super().__init__(
            name="ekf_estimation",
            context=context,
            inputs=["panels"],
            outputs=["estimate"],
        )
        self._tracker: Optional[EkfTracker] = None
        self._height_kf: Optional[HeightKF] = None
        self.is_their_target_prev: bool = False
        self.last_update_time: float = time.perf_counter()

    def set_estimator(self, tracker: EkfTracker) -> None:
        """Inject the tracker (built in the engine's ``initialize`` hook)."""
        self._tracker = tracker
        self.is_their_target_prev = False

    def set_height_kf(self, height_kf: HeightKF) -> None:
        """Inject the height filter (built in the engine's ``initialize`` hook)."""
        self._height_kf = height_kf

    @property
    def tracker(self) -> EkfTracker:
        if self._tracker is None:
            raise RuntimeError(
                "EkfTracker not set. Engine must call set_estimator() in initialize()."
            )
        return self._tracker

    @property
    def height_kf(self) -> HeightKF:
        if self._height_kf is None:
            raise RuntimeError(
                "HeightKF not set. Engine must call set_height_kf() in initialize()."
            )
        return self._height_kf

    def reset(self) -> None:
        """Force re-initialisation on the next observation (target lost)."""
        self.is_their_target_prev = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_panels(panels: List[ArmorPanel]) -> List[ArmorPanel]:
        return [p for p in panels if p.position is not None]

    @staticmethod
    def _closest(valid: List[ArmorPanel]) -> ArmorPanel:
        return min(valid, key=lambda p: float(np.linalg.norm(p.position)))

    # ------------------------------------------------------------------
    # @real
    # ------------------------------------------------------------------

    @real()
    def _run_estimate(
        self, panels: Optional[List[ArmorPanel]]
    ) -> Optional[RobotStateEstimate]:
        """Run one filter iteration; predict-only when there's no fresh observation."""
        # ``panels is None`` == no locked target (engine nulls it on true loss).
        # An empty list can't happen for a locked target, but guard it anyway.
        if not panels:
            self.is_their_target_prev = False
            return None

        radius = self.tracker.radius
        valid = self._valid_panels(panels)

        # First frame after acquiring the target: re-seed both filters at the
        # back-projected centre / observed height of the closest panel so they
        # start near truth instead of crawling from the previous state.
        if not self.is_their_target_prev:
            if not valid:
                self.is_their_target_prev = False
                return None
            seed = self._closest(valid)
            x_obs, y_obs = float(seed.position[0]), float(seed.position[1])
            yaw_obs = float(seed.yaw)
            cx = x_obs - radius * np.cos(yaw_obs)
            cy = y_obs - radius * np.sin(yaw_obs)
            self.tracker.reinit(
                np.array([cx, cy, 0.0, 0.0, yaw_obs, self.tracker.omega0])
            )
            self.height_kf.reinit(float(seed.position[2]))
            self.last_update_time = (
                self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
            )

        self.is_their_target_prev = True

        frame_ts = self.ctx.frame_ts
        if self.ctx.new_observation and valid and frame_ts is not None:
            dt = float(frame_ts - self.last_update_time)
            self.last_update_time = frame_ts
            if dt > 0:
                self.tracker.predict(dt)
            for panel in valid:
                self.tracker.update(
                    float(panel.position[0]),
                    float(panel.position[1]),
                    float(panel.yaw),
                )
            # All panels share one height -> average the visible z's.
            z_meas = float(np.mean([p.position[2] for p in valid]))
            self.height_kf.update(max(dt, 0.0), z_meas)
        else:
            now = time.perf_counter()
            dt = now - self.last_update_time
            assert dt >= 0, (
                f"Negative dt computed in EKF estimation: {dt:.4f}s "
                f"(now={now:.4f}, last_update_time={self.last_update_time:.4f})"
            )
            self.last_update_time = now
            self.tracker.update_with_no_observation(dt)
            self.height_kf.update_with_no_observation(dt)

        # ``confidence`` is informational; downstream only logs it.
        trace_pos = float(self.tracker.P[0, 0] + self.tracker.P[1, 1])
        confidence = 1.0 / (1.0 + trace_pos)

        return RobotStateEstimate(
            value=np.asarray(self.tracker.x, dtype=np.float32).copy(),
            timestamp=self.last_update_time,
            confidence=confidence,
            a_radius=radius,
            b_radius=radius,
            aim_z=float(self.height_kf.estimate()[0]),
        )

    # ------------------------------------------------------------------
    # @mock -- canned estimate so the rest of the pipeline can run on a
    # laptop with no camera / no MCU. The state matches a target hovering
    # 100 cm in front of the turret with no centre drift.
    # ------------------------------------------------------------------

    @mock
    def _run_mock(
        self, panels: Optional[List[ArmorPanel]]
    ) -> Optional[RobotStateEstimate]:
        st = config.spinning_target
        radius_cm = float(st.radius_cm)
        omega0 = float(st.omega_rpm) * 2.0 * np.pi / 60.0
        return RobotStateEstimate(
            value=np.array([0.0, 100.0, 0.0, 0.0, 0.0, omega0], dtype=np.float32),
            timestamp=time.perf_counter(),
            confidence=1.0,
            a_radius=radius_cm,
            b_radius=radius_cm,
            aim_z=0.0,
        )
