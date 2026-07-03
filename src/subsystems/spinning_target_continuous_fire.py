"""Continuous-fire ballistic module for the single spinning target.

Uses a cascade scipy LM solver to compute yaw and pitch toward the predicted
panel position at impact time. The target spins slowly (10 RPM), so we always
continuously fire rather than waiting for panel alignment.

Contract (output yaw in the MCU convention; internal angles from +x CCW -- the
canonical convention lives in ballistics/solver.py):
    inputs:  ``estimate`` (must carry ``aim_z``; ``a_radius`` is the orbit
             radius, same for all arms), ``panels`` (presence gate only)
    outputs: ``solution`` (``is_confident=False`` = straight-line aim, hold fire)

INVARIANT (shared with ekf.py): the target has ``config.spinning_target.n_panels``
identical panels evenly spaced in azimuth, so panel ``k`` sits at
``center + r * [cos(theta + k*2pi/n), sin(theta + k*2pi/n)]`` with a single
radius ``r``. The vertical aim is the filtered ``estimate.aim_z`` (all panels
share one height -- no config z_offset).

This mirrors ``ballistics/full_state_continuous_fire.py`` but for a single
radius and ``n``-fold (not 4-fold) symmetry, and advances the estimate in closed
form (constant velocity + constant spin) instead of via a filter instance.

Input units: estimate state in cm / rad, config in cm/s^2 and cm/s. Internally
converts to metres for the solver.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import root

from src.core.module import Module, mock, real
from src.subsystems.ballistics.solver import (
    make_f_no_spin,
    mcu_yaw_from_xy,
    solve_no_spin,
)
from src.toolbox.globals import config
from src.types.autoaim import (
    ArmorPanel,
    BallisticSolution,
    EkfAutoAimContext,
    RobotStateEstimate,
)

BALLISTIC = config.ballistic
METERS_TO_CM = 100.0
# Sent to embedded in continuous-fire mode; the cv_state field signals the fire
# mode so the alignment time value is unused by firmware in this state.
_CONT_FIRE_ALIGNMENT_TIME_MS = 255


# ---------------------------------------------------------------------------
# N-fold panel geometry (single radius). Generalises solver.py's 4-panel
# helpers; same [cos, sin] convention as the estimator's back-projection.
# ---------------------------------------------------------------------------

def _panel_normals(theta: float, n: int) -> np.ndarray:
    """Outward unit normals of the ``n`` panels, rows ``k = 0..n-1``."""
    angles = theta + np.arange(n) * (2.0 * np.pi / n)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _select_panel(p_t: np.ndarray, tht_t: float, n: int) -> int:
    """Index of the panel whose outward normal best faces the turret (at origin)."""
    facing = -p_t[:2]
    norm = np.linalg.norm(facing)
    if norm < 1e-9:
        return 0
    return int(np.argmax(_panel_normals(tht_t, n) @ (facing / norm)))


def _select_panel_at_time(
    p_t: np.ndarray, v_t: np.ndarray, tht_t: float, omg_t: float, t: float, n: int
) -> int:
    """Like :func:`_select_panel`, for the pose extrapolated ``t`` seconds ahead."""
    p_future = p_t + v_t * t
    tht_future = tht_t + omg_t * t
    return _select_panel(p_future, tht_future, n)


def _cascade_solve(
    b: np.ndarray,
    s: float,
    g: float,
    p_t: np.ndarray,
    v_t: np.ndarray,
    tht_t: float,
    omg_t: float,
    radius: float,
    n_panels: int,
    tol: float = 1e-4,
) -> Dict:
    """Cascade LM solve for [yaw, pitch, travel_time] to hit the best panel.

    Stage 1 solves for the target *centre* (no spin terms) to get a good travel
    time; stage 2 picks the panel facing us at that time and re-solves with the
    panel's orbit offset ``radius * [cos, sin](theta + k*2pi/n)`` rotating at
    omega. Returns a dict with success, yaw (MCU convention), pitch, time,
    panel, residual.
    """
    ns = solve_no_spin(b, s, g, p_t, v_t, tol=tol)

    if ns["success"]:
        panel = _select_panel_at_time(p_t, v_t, tht_t, omg_t, ns["time"], n_panels)
    else:
        panel = _select_panel(p_t, tht_t, n_panels)
    spin_init = np.array([ns["yaw"], ns["pitch"], ns["time"]])
    # MINPACK scales finite-difference steps by |x|, so a denormal-tiny
    # component from an exact stage-1 solve (yaw ~1e-32 for a dead-ahead target)
    # freezes that variable. Snap to exactly 0 (absolute step).
    spin_init[np.abs(spin_init) < 1e-12] = 0.0

    tht_offset = panel * (2.0 * np.pi / n_panels)
    f_centre = make_f_no_spin(b, s, g, p_t, v_t)

    def f_spin(x: np.ndarray) -> np.ndarray:
        t = x[2]
        tht = tht_t + omg_t * t + tht_offset
        residual = f_centre(x)
        # Shift the target from the centre to the orbiting panel.
        residual[0] -= radius * np.cos(tht)
        residual[1] -= radius * np.sin(tht)
        return residual

    result = root(f_spin, spin_init, method="lm", tol=tol)
    residual = float(np.linalg.norm(f_spin(result.x)))
    success = residual < tol

    return {
        "success": success,
        "yaw": float(result.x[0]) if success else None,
        "pitch": float(result.x[1]) if success else None,
        "time": float(result.x[2]) if success else None,
        "panel": panel,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class SpinningTargetContinuousFireModule(Module[EkfAutoAimContext]):
    """Continuous-fire firing solution for the single fixed-radius spinning target.

    Advances the estimate by the processing + actuation delay before solving so
    the bullet intercepts the panel at its future position. Returns
    ``alignment_time_ms=255`` as a dummy; the embedded uses
    ``cv_state=CONTINUOUS_FIRE`` to select continuous fire.
    """

    def __init__(self, context: EkfAutoAimContext):
        super().__init__(
            name="spinning_target_continuous_fire",
            context=context,
            inputs=["estimate", "panels"],
            outputs=["solution"],
        )
        self._g: float = -(abs(BALLISTIC.gravity) / METERS_TO_CM)      # m/s^2, negative
        self._v: float = BALLISTIC.projectile_velocity / METERS_TO_CM  # m/s
        self._barrel: np.ndarray = np.asarray(BALLISTIC.barrel_offset, dtype=float)  # m
        self._lookahead: float = float(BALLISTIC.lookahead_time)       # s
        self._n_panels: int = int(config.spinning_target.n_panels)

    @real()
    def _run_solve(
        self,
        estimate: Optional[RobotStateEstimate],
        panels: Optional[List[ArmorPanel]],
    ) -> Optional[BallisticSolution]:
        """Solve for the firing solution toward the predicted panel position."""
        if estimate is None or not panels:
            return None
        if estimate.aim_z is None:
            self.log.warning("estimate has no aim_z; cannot solve")
            return None

        # Advance the estimate to firing time: processing delay since the
        # estimate + the constant lead offset. Closed-form constant-velocity /
        # constant-spin (no filter instance needed).
        dt = (time.perf_counter() - estimate.timestamp) + self._lookahead
        state = estimate.value
        pred_x = float(state[0] + state[2] * dt)
        pred_y = float(state[1] + state[3] * dt)
        tht_t = float(state[4] + state[5] * dt)
        omg_t = float(state[5])

        p_t = np.array([pred_x / METERS_TO_CM, pred_y / METERS_TO_CM, estimate.aim_z / METERS_TO_CM])
        v_t = np.array([float(state[2]) / METERS_TO_CM, float(state[3]) / METERS_TO_CM, 0.0])
        radius = estimate.a_radius / METERS_TO_CM

        result = _cascade_solve(
            b=self._barrel,
            s=self._v,
            g=self._g,
            p_t=p_t,
            v_t=v_t,
            tht_t=tht_t,
            omg_t=omg_t,
            radius=radius,
            n_panels=self._n_panels,
        )

        if not result["success"]:
            # Out of range / no ballistic arc: keep tracking, hold fire with a
            # straight-line aim at the predicted panel centre.
            self.log.warning(
                "solver failed (residual=%.4f) -- tracking only", result["residual"]
            )
            return BallisticSolution(
                pitch=float(np.arctan2(p_t[2], np.hypot(p_t[0], p_t[1]))),
                yaw=mcu_yaw_from_xy(float(p_t[0]), float(p_t[1])),
                alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
                is_confident=False,
            )

        return BallisticSolution(
            pitch=result["pitch"],
            yaw=result["yaw"],
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
            is_confident=True,
        )

    @mock
    def _run_mock_solve(
        self,
        estimate: Optional[RobotStateEstimate],
        panels: Optional[List[ArmorPanel]],
    ) -> Optional[BallisticSolution]:
        return BallisticSolution(
            pitch=float(np.deg2rad(10)),
            yaw=float(np.deg2rad(20)),
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
            is_confident=True,
        )
