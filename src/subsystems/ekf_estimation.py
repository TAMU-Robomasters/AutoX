"""EKF estimation module for the spinning-target auto-aim pipeline.

Wraps :class:`EkfTracker` in a :class:`Module` so it slots into the engine's
clipboard-style ``Context`` plumbing. Mirrors
:class:`ParticleFilterEstimationModule` (``src/subsystems/pf.py``) but
consumes ``panels`` directly -- there is no classification/targeting layer
because the rig has a single, single-radius spinning target.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.ekf import EkfTracker
from src.toolbox.globals import config
from src.types.autoaim import (
    ArmorPanel,
    EkfAutoAimContext,
    RobotStateEstimate,
)


def _default_ekf_tracker() -> EkfTracker:
    """Build an ``EkfTracker`` from ``config.spinning_target``."""
    radius_cm = float(config.spinning_target.radius_cm)
    omega_rpm = float(config.spinning_target.omega_rpm)
    omega0 = omega_rpm * 2.0 * np.pi / 60.0
    return EkfTracker(radius=radius_cm, omega0=omega0)


class EkfEstimationModule(Module[EkfAutoAimContext]):
    """Estimate spinning-target state with a linear KF (back-projection trick)."""

    def __init__(self, context: EkfAutoAimContext):
        super().__init__(
            name="ekf_estimation",
            context=context,
            inputs=["panels"],
            outputs=["estimate"],
        )
        self._tracker: Optional[EkfTracker] = None
        self.is_their_target_prev: bool = False
        self.last_update_time: float = time.perf_counter()

    def set_estimator(self, tracker: EkfTracker) -> None:
        """Inject the tracker (built in the engine's ``initialize`` hook)."""
        self._tracker = tracker
        self.is_their_target_prev = False

    @property
    def tracker(self) -> EkfTracker:
        if self._tracker is None:
            raise RuntimeError(
                "EkfTracker not set. Engine must call set_estimator() in initialize()."
            )
        return self._tracker

    def reset(self) -> None:
        """Force re-initialisation on the next observation."""
        self.is_their_target_prev = False

    # ------------------------------------------------------------------
    # @real
    # ------------------------------------------------------------------

    @real()
    def _run_estimate(
        self, panels: Optional[List[ArmorPanel]]
    ) -> Optional[RobotStateEstimate]:
        """Run one filter iteration on the visible panels."""
        if self._tracker is None:
            raise RuntimeError(
                "EkfTracker not set. Engine must call set_estimator() in initialize()."
            )

        if not panels:
            self.is_their_target_prev = False
            return None

        radius = self.tracker.radius

        # First frame a panel becomes visible: re-seed the tracker at the
        # back-projected centre of the first valid panel so the filter starts
        # close to truth instead of crawling from the origin.
        if not self.is_their_target_prev:
            seed_panel = next(
                (p for p in panels if p.position is not None), None
            )
            if seed_panel is None:
                self.is_their_target_prev = False
                return None
            x_obs, y_obs = float(seed_panel.position[0]), float(seed_panel.position[1])
            yaw_obs = float(seed_panel.yaw)
            cx = x_obs - radius * np.cos(yaw_obs)
            cy = y_obs - radius * np.sin(yaw_obs)
            self.tracker.reinit(
                np.array([cx, cy, 0.0, 0.0, yaw_obs, self.tracker.omega0])
            )
            self.last_update_time = (
                self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
            )

        self.is_their_target_prev = True

        if self.ctx.new_observation:
            dt = float(self.ctx.frame_ts - self.last_update_time)
            self.last_update_time = self.ctx.frame_ts
            if dt > 0:
                self.tracker.predict(dt)
            for panel in panels:
                if panel.position is None:
                    continue
                self.tracker.update(
                    float(panel.position[0]),
                    float(panel.position[1]),
                    float(panel.yaw),
                )
        else:
            now = time.perf_counter()
            dt = now - self.last_update_time
            assert dt >= 0, (
                f"Negative dt computed in EKF estimation: {dt:.4f}s "
                f"(now={now:.4f}, last_update_time={self.last_update_time:.4f})"
            )
            self.last_update_time = now
            self.tracker.update_with_no_observation(dt)

        # ``confidence`` is informational; downstream only logs it.
        trace_pos = float(self.tracker.P[0, 0] + self.tracker.P[1, 1])
        confidence = 1.0 / (1.0 + trace_pos)

        return RobotStateEstimate(
            value=np.asarray(self.tracker.x, dtype=np.float32).copy(),
            timestamp=self.last_update_time,
            confidence=confidence,
            a_radius=radius,
            b_radius=radius,
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
        radius_cm = float(config.spinning_target.radius_cm)
        omega0 = float(config.spinning_target.omega_rpm) * 2.0 * np.pi / 60.0
        return RobotStateEstimate(
            value=np.array([0.0, 100.0, 0.0, 0.0, 0.0, omega0], dtype=np.float32),
            timestamp=time.perf_counter(),
            confidence=1.0,
            a_radius=radius_cm,
            b_radius=radius_cm,
        )
