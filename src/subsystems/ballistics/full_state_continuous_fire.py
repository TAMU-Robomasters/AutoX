"""Full-state continuous-fire ballistic module.

Uses a cascade scipy LM solver to compute yaw and pitch toward the predicted
panel position at impact time. Intended for slow-spinning or stationary targets
where continuous fire is preferred over waiting for panel alignment.

Contract (output yaw in the MCU convention; internal angles from +x CCW -- the
canonical convention lives in ballistics/solver.py):
    inputs:  ``estimate`` (must carry ``aim_z`` and per-parity radii),
             ``target_robot`` (presence gate only)
    outputs: ``solution`` (``is_confident=False`` = straight-line aim, hold fire)

INVARIANT (shared with kf.py and panel_tracking.py): panel ``k`` sits at angle
``theta + k*90deg``; even ``k`` uses ``estimate.a_radius``, odd ``k`` uses
``estimate.b_radius``. The vertical aim is ``estimate.aim_z`` (live panel
mid-height, cm -- no config z_offset).

Input units: estimate state in cm / rad, config in cm/s^2 and cm/s.
Internally converts to metres for the solver.
"""
import time
from typing import Dict, Optional

import numpy as np
from scipy.optimize import root

from src.core.module import Module, mock, real
from src.subsystems.ballistics.solver import (
    make_f_no_spin,
    mcu_yaw_from_xy,
    select_panel,
    select_panel_at_time,
    solve_no_spin,
)
from src.subsystems.estimation.filters import FullStateKF
from src.toolbox.globals import config
from src.types.autoaim import (
    BallisticSolution,
    EnemyRobot,
    FullStateAutoAimContext,
    RobotStateEstimate,
)

BALLISTIC = config.ballistic
METERS_TO_CM = 100.0
# Sent to embedded in continuous-fire mode; the cv_state field signals the fire
# mode so the alignment time value is unused by firmware in this state.
_CONT_FIRE_ALIGNMENT_TIME_MS = 255


def _cascade_solve(
    b: np.ndarray,
    s: float,
    g: float,
    p_t: np.ndarray,
    v_t: np.ndarray,
    tht_t: float,
    omg_t: float,
    a_radius: float,
    b_radius: float,
    tol: float = 1e-4,
) -> Dict:
    """Cascade LM solve for [yaw, pitch, travel_time] to hit the best panel.

    Stage 1 solves for the robot *centre* (no spin terms) to get a good travel
    time; stage 2 picks the panel facing us at that time and re-solves with the
    panel's orbit offset ``r * [cos, sin](theta + k*90deg)`` rotating at omega.

    Args:
        b: Barrel tip offset [bx, by, bz] in metres.
        s: Projectile speed in m/s.
        g: Gravity in m/s^2 (negative, e.g. -9.81).
        p_t: Robot centre position [x, y, z] in metres at firing time.
        v_t: Robot centre velocity [vx, vy, vz] in m/s.
        tht_t: Robot heading (rad, from +x CCW) at firing time.
        omg_t: Robot angular velocity (rad/s, CCW-positive) at firing time.
        a_radius: Orbit radius of even-parity panels (0 & 2) in metres.
        b_radius: Orbit radius of odd-parity panels (1 & 3) in metres.
        tol: Residual convergence tolerance.

    Returns:
        dict with keys: success, yaw (MCU convention), pitch, time, panel, residual.
    """
    ns = solve_no_spin(b, s, g, p_t, v_t, tol=tol)

    if ns["success"]:
        panel = select_panel_at_time(p_t, v_t, tht_t, omg_t, ns["time"])
    else:
        panel = select_panel(p_t, tht_t)
    spin_init = np.array([ns["yaw"], ns["pitch"], ns["time"]])
    # MINPACK scales finite-difference steps by |x|, so a denormal-tiny
    # component from an exact stage-1 solve (yaw ~1e-32 for a dead-ahead
    # target) freezes that variable. Snap to exactly 0 (absolute step).
    spin_init[np.abs(spin_init) < 1e-12] = 0.0

    r = a_radius if panel % 2 == 0 else b_radius
    tht_offset = panel * (np.pi / 2.0)

    f_centre = make_f_no_spin(b, s, g, p_t, v_t)

    def f_spin(x: np.ndarray) -> np.ndarray:
        t = x[2]
        tht = tht_t + omg_t * t + tht_offset
        residual = f_centre(x)
        # Shift the target from the centre to the orbiting panel.
        residual[0] -= r * np.cos(tht)
        residual[1] -= r * np.sin(tht)
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


class FullStateContinuousFireModule(Module[FullStateAutoAimContext]):
    """Continuous-fire firing solution using the full cascade ballistics solver.

    For slow/non-spinning targets. Advances the estimator state by the
    processing delay before solving so the bullet intercepts the target at the
    correct future position. Returns alignment_time_ms=255 as a dummy value;
    the embedded uses cv_state=CONTINUOUS_FIRE to select continuous fire mode
    rather than gating on alignment time.
    """

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="full_state_continuous_fire",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["solution"],
        )
        self._g: float = -(abs(BALLISTIC.gravity) / METERS_TO_CM)    # m/s^2, negative
        self._v: float = BALLISTIC.projectile_velocity / METERS_TO_CM  # m/s
        self._barrel: np.ndarray = np.asarray(BALLISTIC.barrel_offset, dtype=float)  # m

    @real()
    def _run_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Solve for the firing solution toward the predicted panel position."""
        if estimate is None or target_robot is None:
            return None
        if estimate.aim_z is None:
            # Engine must not route here before parity anchoring sets aim_z.
            self.log.warning("estimate has no aim_z; cannot solve")
            return None

        # Advance state to firing time (processing delay since the estimate) via
        # constant-velocity extrapolation -- no estimator instance needed.
        time_since_estimate = time.perf_counter() - estimate.timestamp
        pred = FullStateKF.predict_ahead(estimate.value, time_since_estimate + BALLISTIC.lookahead_time)
        self.log.debug("predicted_state %.2f %.2f %.2f", pred[0], pred[1], estimate.aim_z)
        p_t = np.array(
            [pred[0] / METERS_TO_CM, pred[1] / METERS_TO_CM, estimate.aim_z / METERS_TO_CM]
        )
        v_t = np.array([pred[2] / METERS_TO_CM, pred[3] / METERS_TO_CM, 0.0])
        tht_t = float(pred[4])
        omg_t = float(pred[5])

        result = _cascade_solve(
            b=self._barrel,
            s=self._v,
            g=self._g,
            p_t=p_t,
            v_t=v_t,
            tht_t=tht_t,
            omg_t=omg_t,
            a_radius=estimate.a_radius / METERS_TO_CM,
            b_radius=estimate.b_radius / METERS_TO_CM,
        )

        if not result["success"]:
            # Out of range / no ballistic arc: keep tracking, hold fire.
            self.log.warning("solver failed (residual=%.4f) -- tracking only", result["residual"])
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
        )

    @mock
    def _run_mock_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Fixed plausible solution for mock mode."""
        return BallisticSolution(
            pitch=np.deg2rad(10),
            yaw=np.deg2rad(20),
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
        )
