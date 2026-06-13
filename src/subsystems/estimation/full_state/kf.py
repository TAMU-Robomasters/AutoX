"""Kalman-filter estimation module.

Feeds the target robot's tracked panel observation into a ``FullStateKF`` (a
pair of constant-velocity Kalman filters -- ``PositionKF`` for ``[x, y, vx,
vy]`` and ``AngleKF`` for ``[theta, omega]``) and emits a
``RobotStateEstimate``.

Contract (units/frames: turret frame, cm, angles from +x CCW -- see
src/types/autoaim.py):
    inputs:  ``target_robot`` (panels transformed to the turret frame; panel
             ids optionally assigned by PanelTrackingModule)
    outputs: ``estimate`` (6-D [x, y, vx, vy, theta, omega] + per-parity radii
             + aim_z)

The back-projection radius is chosen per measurement from the observed panel's
id parity (``set_panel_radii``); ids 0&2 use ``r_even``, 1&3 use ``r_odd``,
unknown ids use the mean. INVARIANT (shared with PanelTrackingModule and
ballistics/solver.py): panel id ``k`` sits at angle ``theta + k*90deg``, so id
parity == the parity of ``k`` in ``panel_normals``; ``a_radius`` is stamped
from ``r_even`` and ``b_radius`` from ``r_odd``.

Statefulness/reset: auto-reinits from the first panel whenever the target was
lost on the previous run; the engine can force this via ``reset()`` or seed
explicitly via ``reinit_from_panel`` (state 2a -> 2b anchoring). The module
never reads config or disk -- the engine injects everything.
"""
import time
from typing import Optional

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.estimation.full_state.kalman_filter import (
    AngleKF,
    FullStateKF,
    HeightKF,
    PositionKF,
)
from src.toolbox.globals import config
from src.types.autoaim import (
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
    RobotStateEstimate,
)


def _default_full_state_kf(r: float = 23.5) -> FullStateKF:
    """Create a FullStateKF with the same hyperparameters as ``_default_particle_filter``."""
    position_kf = PositionKF(
        r_pos=30.0,
        q_vx=100.0,
        q_vy=100.0,
        r=r,
        init_std=(100.0, 100.0, 10.0, 10.0),
    )
    angle_kf = AngleKF(
        N=4,
        r_yaw=np.radians(8.0),
        q_omega=0.6,
        init_std_theta=0.3,
        init_std_omega=15.0,
    )
    height_kf = HeightKF(
        q=float(config.estimation.height_kf_q),
        r=float(config.estimation.height_kf_r),
    )
    return FullStateKF(position_kf, angle_kf, height_kf)


class KalmanFilterEstimationModule(Module[FullStateAutoAimContext]):
    """Estimate robot state using a FullStateKF (position KF + angle KF)."""

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="kalman_filter_estimation",
            context=context,
            inputs=["target_robot"],
            outputs=["estimate"],
        )
        self._estimator: Optional[FullStateKF] = None
        self.is_their_target_prev = False
        self.is_their_target = False
        self.last_update_time = time.perf_counter()

        # Per-parity orbit radii (cm). Engine-injected: the state-1 initial
        # guess via set_panel_radii(guess, guess), the anchored values at 2a->2b.
        self._r_even: float = 23.5
        self._r_odd: float = 23.5
        # Aim geometry (engine-injected at anchoring; None in state 1).
        self._height_delta: Optional[float] = None
        self._low_parity: int = 0

    def set_estimator(self, estimator: FullStateKF) -> None:
        """Inject the FullStateKF instance created by the engine."""
        self._estimator = estimator

    @property
    def estimator(self) -> FullStateKF:
        """Expose the underlying estimator (used by engine for reinit)."""
        if self._estimator is None:
            raise RuntimeError("Estimator is not set. Engine must inject it in initialize()")
        return self._estimator

    def set_panel_radii(self, r_even: float, r_odd: float) -> None:
        """Set the per-parity orbit radii (cm) used for back-projection + stamping."""
        self._r_even = float(r_even)
        self._r_odd = float(r_odd)

    def set_aim_geometry(self, height_delta: float, low_parity: int) -> None:
        """Set the inter-pair height delta (cm) and which parity is the LOW pair.

        Enables ``aim_z`` on the output: the mid-height between the pairs,
        derived from the live tracked panel's z.
        """
        self._height_delta = float(height_delta)
        self._low_parity = int(low_parity)

    def clear_aim_geometry(self) -> None:
        """Disable aim_z (back to state-1 behavior)."""
        self._height_delta = None

    def reset(self) -> None:
        """Force re-initialisation on the next observation."""
        self.is_their_target_prev = False

    def reinit_from_panel(self, panel: ArmorPanel) -> None:
        """Explicitly re-seed the filter from one panel (state 2a -> 2b anchoring).

        Same prior shape as the auto-reinit: the position component starts at
        the panel itself and the first update's back-projection pulls it to the
        robot centre. Suppresses the auto-reinit on the next run.
        """
        assert panel.position is not None
        z0 = self._aim_z_for(panel)
        if z0 is None:
            z0 = float(panel.position[2])
        prior = np.array(
            [panel.position[0], panel.position[1], 0, 0, panel.yaw, 0, z0],
            dtype=np.float32,
        )
        self.estimator.reinit(prior)
        self.is_their_target_prev = True
        self.last_update_time = (
            self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()
        )

    def _radius_for(self, panel: ArmorPanel) -> float:
        """Back-projection radius for this panel's id parity (mean if unknown)."""
        if panel.id is None:
            return (self._r_even + self._r_odd) / 2.0
        return self._r_even if panel.id % 2 == 0 else self._r_odd

    def _aim_z_for(self, panel: ArmorPanel) -> Optional[float]:
        """Mid-height between the pairs, from the live panel z (None in state 1).

        Relative to the observed panel so a robot on a platform aims correctly:
        from the LOW pair aim up by height_delta/2, from the high pair aim down.
        """
        if self._height_delta is None or panel.position is None:
            return None
        panel_z = float(panel.position[2])
        if panel.id is None:
            return panel_z  # transient: id missing, aim at the panel itself
        sign = 1.0 if panel.id % 2 == self._low_parity else -1.0
        return panel_z + sign * self._height_delta / 2.0

    @staticmethod
    def _select_panel(target_robot: EnemyRobot) -> Optional[ArmorPanel]:
        """Closest panel, preferring panels that carry a tracked id."""
        candidates = [
            (float(np.linalg.norm(p.position)), i, p)
            for i, p in enumerate(target_robot.panels)
            if p.position is not None
        ]
        if not candidates:
            return None
        with_id = [c for c in candidates if c[2].id is not None]
        pool = with_id or candidates
        return min(pool)[2]

    @real()
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RobotStateEstimate]:
        """Run one filter iteration on the target robot's tracked panel."""
        if target_robot is None:
            self.is_their_target_prev = False
            return None

        self.is_their_target = True

        panel = self._select_panel(target_robot)
        # _select_panel only returns panels with a position.
        position = panel.position if panel is not None else None
        if not self.is_their_target_prev and panel is not None and position is not None:
            z0 = self._aim_z_for(panel)
            if z0 is None:
                z0 = float(position[2])
            prior = np.array(
                [position[0], position[1], 0, 0, panel.yaw, 0, z0],
                dtype=np.float32,
            )
            self.estimator.reinit(prior)
            # Reset clock so first dt isn't stale from before target acquisition
            self.last_update_time = self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()

        self.is_their_target_prev = True
        frame_ts = self.ctx.frame_ts
        if self.ctx.new_observation and panel is not None and position is not None and frame_ts is not None:
            measurement = np.array(
                [[position[0], position[1], panel.yaw]],
                dtype=np.float32,
            )  # shape (1, 3)
            dt = frame_ts - self.last_update_time
            self.last_update_time = frame_ts
            estimate, confidence = self.estimator.update(
                dt, measurement, r=self._radius_for(panel), z_meas=self._aim_z_for(panel)
            )
        else:
            current_time = time.perf_counter()
            dt = current_time - self.last_update_time
            assert dt >= 0, f"Negative dt computed in estimation module: {dt:.4f}s (current_time={current_time:.4f}, last_update_time={self.last_update_time:.4f})"

            self.last_update_time = current_time
            estimate, confidence = self.estimator.update_with_no_observation(dt)

        # aim_z is the FILTERED panel-center height (index 6), exposed only once
        # the engine has anchored the inter-pair height geometry (state 2b);
        # None in state 1 preserves the single-panel aiming path.
        aim_z = float(estimate[6]) if self._height_delta is not None else None

        return RobotStateEstimate(
            value=estimate,
            timestamp=self.last_update_time,
            confidence=float(confidence),
            a_radius=self._r_even,
            b_radius=self._r_odd,
            aim_z=aim_z,
        )

    @mock
    def _run_mock_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RobotStateEstimate]:
        """Fixed plausible estimate for mock mode."""
        return RobotStateEstimate(
            value=np.array([0.0, 200.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            timestamp=time.perf_counter(),
            confidence=1.0,
            a_radius=self._r_even,
            b_radius=self._r_odd,
            aim_z=None,
        )
