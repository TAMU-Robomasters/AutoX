"""Single-panel xy position+velocity estimation (state 1 / state 2a aiming).

Contract (turret frame, cm, cm/s -- see src/types/autoaim.py):
    inputs:  ``target_robot`` (panels transformed to the turret frame)
    outputs: ``xy_estimate`` (``PanelEstimate``: the tracked *panel itself*,
             not the robot centre -- [x, y, vx, vy] + raw panel z)

Used while the robot's geometry is unknown (state 1) or un-anchored (state
2a): track the closest panel with a plain constant-velocity KF and aim at it
directly. Reuses ``PositionKF`` with orbit radius r=0, which degenerates the
back-projection to identity (the measurement IS the panel position).

Track switching: the filter re-seeds when the tracked panel's id changes
(when ids are available) or, with no ids, when the closest panel jumps more
than ``jump_reinit_cm`` (a different panel rotated into view). The engine can
force a re-seed via ``reset()`` on target loss.
"""

import time
from typing import Optional

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.estimation.full_state.kalman_filter import PositionKF
from src.types.autoaim import (
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
    PanelEstimate,
)


def _default_panel_kf() -> PositionKF:
    """Constant-velocity KF over the panel itself (r=0: no back-projection)."""
    return PositionKF(
        r_pos=3.0,        # cm, single-frame panel position noise
        q_vx=100.0,
        q_vy=100.0,
        r=0.0,            # back_project(x, y, yaw, 0) == (x, y)
        init_std=(10.0, 10.0, 50.0, 50.0),
    )


class SinglePanelEstimationModule(Module[FullStateAutoAimContext]):
    """Track the closest panel's [x, y, vx, vy] with a plain CV Kalman filter."""

    def __init__(self, context: FullStateAutoAimContext, jump_reinit_cm: float = 15.0):
        super().__init__(
            name="single_panel_estimation",
            context=context,
            inputs=["target_robot"],
            outputs=["xy_estimate"],
        )
        self._kf = _default_panel_kf()
        self._jump_reinit_cm = float(jump_reinit_cm)
        self._initialized = False
        self._last_id: Optional[int] = None
        self._last_xy: Optional[np.ndarray] = None
        self._last_z: float = 0.0
        self.last_update_time = time.perf_counter()

    def reset(self) -> None:
        """Force a re-seed on the next observation (target lost / state change)."""
        self._initialized = False
        self._last_id = None
        self._last_xy = None

    def _needs_reinit(self, panel: ArmorPanel, xy: np.ndarray) -> bool:
        if not self._initialized:
            return True
        if panel.id is not None and self._last_id is not None:
            return panel.id != self._last_id
        if self._last_xy is not None:
            return bool(np.linalg.norm(xy - self._last_xy) > self._jump_reinit_cm)
        return False

    @staticmethod
    def _closest_panel(target_robot: EnemyRobot) -> Optional[ArmorPanel]:
        candidates = [
            (float(np.linalg.norm(p.position)), i, p)
            for i, p in enumerate(target_robot.panels)
            if p.position is not None
        ]
        if not candidates:
            return None
        return min(candidates)[2]

    @real()
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[PanelEstimate]:
        if target_robot is None:
            self.reset()
            return None

        panel = self._closest_panel(target_robot)

        position = panel.position if panel is not None else None
        if self.ctx.new_observation and panel is not None and position is not None:
            xy = np.asarray(position, dtype=float).flatten()[:2]
            frame_ts = self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
            if self._needs_reinit(panel, xy):
                self._kf.reinit(xy[0], xy[1])
                self.last_update_time = frame_ts
            dt = max(frame_ts - self.last_update_time, 0.0)
            self.last_update_time = frame_ts
            estimate, _ = self._kf.update(dt, xy[0], xy[1], panel.yaw)  # r=0: yaw unused
            self._initialized = True
            self._last_id = panel.id
            self._last_xy = xy
            self._last_z = float(position[2])
        else:
            if not self._initialized:
                return None
            current_time = time.perf_counter()
            dt = max(current_time - self.last_update_time, 0.0)
            self.last_update_time = current_time
            estimate, _ = self._kf.update_with_no_observation(dt)

        return PanelEstimate(
            value=np.asarray(estimate, dtype=np.float32),
            z=self._last_z,
            panel_id=self._last_id,
            timestamp=self.last_update_time,
        )

    @mock
    def _run_mock_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[PanelEstimate]:
        """Fixed plausible panel track for mock mode."""
        return PanelEstimate(
            value=np.array([0.0, 200.0, 0.0, 0.0], dtype=np.float32),
            z=-20.0,
            panel_id=None,
            timestamp=time.perf_counter(),
        )
