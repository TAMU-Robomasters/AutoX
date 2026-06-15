"""Single-panel ballistic module (state 1 / state 2a aiming).

Solves the no-target-acceleration projectile equations (``f_no_spin`` from
ballistics/solver.py -- the full equations with the 1/2*a*t^2 terms dropped)
toward the tracked panel itself: constant-velocity xy from the single-panel
KF, and the panel's *actual measured z* (no config z_offset, no robot-geometry
knowledge needed).

Contract (xy_estimate in turret-frame cm; solver runs in metres; output yaw in
the MCU convention):
    inputs:  ``xy_estimate`` (PanelEstimate from SinglePanelEstimationModule)
    outputs: ``solution`` (``is_confident=False`` = straight-line aim, hold fire)

Validity gates (config.ballistic): horizontal range < ``max_range``, solver
residual < ``solver_tol``, time of flight in (0, ``max_time_of_flight``).
Outside them the module still emits a straight-line tracking aim so the turret
stays on target, with ``is_confident=False`` so the engine reports
``CVState.NO_TARGET`` and the firmware holds fire.
"""

import time
from typing import Optional

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.ballistics.solver import mcu_yaw_from_xy, solve_no_spin
from src.subsystems.estimation.full_state.kalman_filter import FullStateKF
from src.toolbox.globals import config
from src.types.autoaim import (
    BallisticSolution,
    FullStateAutoAimContext,
    PanelEstimate,
)

BALLISTIC = config.ballistic
METERS_TO_CM = 100.0
_ALIGNMENT_TIME_MS = 255  # continuous fire: firmware keys off cv_state, not this


class SinglePanelBallisticModule(Module[FullStateAutoAimContext]):
    """Continuous-fire solution aimed at the tracked panel (no robot geometry)."""

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="single_panel_ballistics",
            context=context,
            inputs=["xy_estimate"],
            outputs=["solution"],
        )
        self._g: float = -(abs(BALLISTIC.gravity) / METERS_TO_CM)      # m/s^2
        self._v: float = BALLISTIC.projectile_velocity / METERS_TO_CM  # m/s
        self._barrel: np.ndarray = np.asarray(BALLISTIC.barrel_offset, dtype=float)  # m
        self._max_tof: float = float(BALLISTIC.max_time_of_flight)     # s
        self._max_range: float = float(BALLISTIC.max_range)            # cm
        self._tol: float = float(BALLISTIC.solver_tol)
        self._lookahead: float = float(BALLISTIC.prediction_lookahead)  # s

    def _track_only(self, p_t_m: np.ndarray) -> BallisticSolution:
        """Straight-line aim with fire held (target out of solvable range)."""
        return BallisticSolution(
            pitch=float(np.arctan2(p_t_m[2], np.hypot(p_t_m[0], p_t_m[1]))),
            yaw=mcu_yaw_from_xy(float(p_t_m[0]), float(p_t_m[1])),
            alignment_time_ms=_ALIGNMENT_TIME_MS,
            is_confident=False,
        )

    @real()
    def _run_solve(
        self, xy_estimate: Optional[PanelEstimate]
    ) -> Optional[BallisticSolution]:
        if xy_estimate is None:
            return None

        # Constant-velocity advance from estimate time to now, plus the
        # configured lead time (latency compensation), in cm.
        dt = max(time.perf_counter() - xy_estimate.timestamp, 0.0) + self._lookahead
        x, y, vx, vy = (float(v) for v in FullStateKF.predict_state(xy_estimate.value, dt))

        p_t = np.array([x, y, xy_estimate.z]) / METERS_TO_CM  # metres
        v_t = np.array([vx, vy, 0.0]) / METERS_TO_CM

        if np.hypot(x, y) > self._max_range:
            self.log.warning("panel out of range (%.0fcm) -- tracking only", np.hypot(x, y))
            return self._track_only(p_t)

        result = solve_no_spin(self._barrel, self._v, self._g, p_t, v_t, tol=self._tol)
        if not result["success"] or not (0.0 < result["time"] < self._max_tof):
            self.log.warning(
                "no valid solution (residual=%.4f, t=%.2fs) -- tracking only",
                result["residual"],
                result["time"],
            )
            return self._track_only(p_t)

        return BallisticSolution(
            pitch=result["pitch"],
            yaw=result["yaw"],
            alignment_time_ms=_ALIGNMENT_TIME_MS,
        )

    @mock
    def _run_mock_solve(
        self, xy_estimate: Optional[PanelEstimate]
    ) -> Optional[BallisticSolution]:
        """Fixed plausible solution for mock mode."""
        return BallisticSolution(
            pitch=np.deg2rad(10),
            yaw=np.deg2rad(20),
            alignment_time_ms=_ALIGNMENT_TIME_MS,
        )
