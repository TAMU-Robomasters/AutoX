"""Full-state continuous-fire ballistic module.

Uses a cascade scipy LM solver to compute yaw and pitch toward the predicted
panel position at impact time. Intended for slow-spinning or stationary targets
where continuous fire is preferred over waiting for panel alignment.

Coordinate frame (right-handed, matches ballitics.py convention):
  x  →  right
  y  →  forward / out (depth)
  z  →  up

Input units: estimate state in cm / rad, config in cm/s^2 and cm/s.
Internally converts to metres for the solver.
"""
from src.core.driver import mock

import time
from typing import Optional

import numpy as np
from scipy.optimize import root

from src.core.module import Module, real
from src.subsystems.particle_filter import ParticleFilter
from src.toolbox.globals import config
from src.types.autoaim import (
    BallisticSolution,
    ParticleFilterAutoAimContext,
    RobotStateEstimate,
    EnemyRobot,
)


BALLISTIC = config.ballistic
METERS_TO_CM = 100.0
# Sent to embedded in continuous-fire mode; the cv_state field signals the fire
# mode so the alignment time value is unused by firmware in this state.
_CONT_FIRE_ALIGNMENT_TIME_MS = 255


# ---------------------------------------------------------------------------
# Solver internals (cascade LM, ported from engines/ballitics.py)
# ---------------------------------------------------------------------------

def _init_guess(p_t: np.ndarray, speed: float, gravity: float) -> np.ndarray:
    g = abs(gravity)
    x = np.hypot(p_t[0], p_t[1])
    y = p_t[2]

    if x < 1e-9:
        t0 = abs(y) / speed if abs(y) > 1e-9 else 0.1
        return np.array([0.0, np.sign(y) * np.pi / 2, t0])

    discriminant = speed**4 - g * (g * x**2 + 2.0 * y * speed**2)
    if discriminant >= 0:
        pitch0 = np.arctan((speed**2 - np.sqrt(discriminant)) / (g * x))
    else:
        pitch0 = np.pi / 4

    cos_p = np.cos(pitch0)
    t0 = x / (speed * cos_p) if abs(cos_p) > 1e-6 else np.linalg.norm(p_t) / speed

    yaw0 = np.arccos(np.clip(p_t[1] / x, -1.0, 1.0))
    if p_t[0] > 0:
        yaw0 = -yaw0

    return np.array([yaw0, pitch0, t0])


def _panel_normals(tht: float) -> np.ndarray:
    return np.array([
        [np.sin(tht),             np.cos(tht)],
        [np.sin(tht + np.pi/2),  np.cos(tht + np.pi/2)],
        [np.sin(tht + np.pi),    np.cos(tht + np.pi)],
        [np.sin(tht - np.pi/2),  np.cos(tht - np.pi/2)],
    ])


def _select_panel(p_t: np.ndarray, tht_t: float) -> int:
    p_xy = -p_t[:2]
    norm = np.linalg.norm(p_xy)
    if norm < 1e-9:
        return 0
    return int(np.argmax(_panel_normals(tht_t) @ (p_xy / norm)))


def _select_panel_at_time(
    p_t: np.ndarray, v_t: np.ndarray, tht_t: float, omg_t: float, t: float
) -> int:
    p_future = p_t + v_t * t
    tht_future = tht_t + omg_t * t
    p_xy = -p_future[:2]
    norm = np.linalg.norm(p_xy)
    if norm < 1e-9:
        return 0
    return int(np.argmax(_panel_normals(tht_future) @ (p_xy / norm)))


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
) -> dict:
    """Cascade LM solve for [yaw, pitch, travel_time] to hit the best panel.

    Args:
        b: Barrel tip offset [bx, by, bz] in metres.
        s: Projectile speed in m/s.
        g: Gravity in m/s^2 (negative, e.g. -9.81).
        p_t: Robot centre position [x, y, z] in metres at firing time.
        v_t: Robot centre velocity [vx, vy, vz] in m/s.
        tht_t: Robot heading (rad) at firing time.
        omg_t: Robot angular velocity (rad/s) at firing time.
        a_radius: Panel orbit radius for panels 0 and 2 (forward/back) in metres.
        b_radius: Panel orbit radius for panels 1 and 3 (left/right) in metres.
        tol: Residual convergence tolerance.

    Returns:
        dict with keys: success, yaw, pitch, time, panel, residual.
    """
    x0 = _init_guess(p_t, s, g)

    def f_no_spin(x):
        yaw, pitch, t = x
        return np.array([
            b[0]*np.cos(yaw) - np.sin(yaw)*(b[1]*np.cos(pitch) - b[2]*np.sin(pitch))
                - s*t*np.sin(yaw)*np.cos(pitch) - p_t[0] - v_t[0]*t,
            b[0]*np.sin(yaw) + np.cos(yaw)*(b[1]*np.cos(pitch) - b[2]*np.sin(pitch))
                + s*t*np.cos(yaw)*np.cos(pitch) - p_t[1] - v_t[1]*t,
            b[1]*np.sin(pitch) + b[2]*np.cos(pitch) + s*t*np.sin(pitch)
                + 0.5*g*t**2 - p_t[2] - v_t[2]*t,
        ])

    ns_res = root(f_no_spin, x0, method='lm', tol=tol)
    ns_residual = float(np.linalg.norm(f_no_spin(ns_res.x)))
    ns_ok = ns_residual < tol

    if ns_ok:
        panel = _select_panel_at_time(p_t, v_t, tht_t, omg_t, float(ns_res.x[2]))
        spin_init = ns_res.x
    else:
        panel = _select_panel(p_t, tht_t)
        spin_init = x0

    r = a_radius if panel in (0, 2) else b_radius
    tht_offset = panel * (np.pi / 2.0)

    def f_spin(x):
        yaw, pitch, t = x
        tht = tht_t + omg_t * t
        return np.array([
            b[0]*np.cos(yaw) - np.sin(yaw)*(b[1]*np.cos(pitch) - b[2]*np.sin(pitch))
                - s*t*np.sin(yaw)*np.cos(pitch) - p_t[0] - v_t[0]*t
                - r*np.sin(tht + tht_offset),
            b[0]*np.sin(yaw) + np.cos(yaw)*(b[1]*np.cos(pitch) - b[2]*np.sin(pitch))
                + s*t*np.cos(yaw)*np.cos(pitch) - p_t[1] - v_t[1]*t
                - r*np.cos(tht + tht_offset),
            b[1]*np.sin(pitch) + b[2]*np.cos(pitch) + s*t*np.sin(pitch)
                + 0.5*g*t**2 - p_t[2] - v_t[2]*t,
        ])

    result = root(f_spin, spin_init, method='lm', tol=tol)
    residual = float(np.linalg.norm(f_spin(result.x)))
    success = residual < tol

    return {
        'success':  success,
        'yaw':      float(result.x[0]) if success else None,
        'pitch':    float(result.x[1]) if success else None,
        'time':     float(result.x[2]) if success else None,
        'panel':    panel,
        'residual': residual,
    }


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class FullStateContinuousFireModule(Module[ParticleFilterAutoAimContext]):
    """Continuous-fire firing solution using the full cascade ballistics solver.

    For slow/non-spinning targets. Advances the particle-filter state by
    (processing delay + feeder delay) before solving so the bullet intercepts
    the target at the correct future position. Returns alignment_time_ms=255
    as a dummy value; the embedded uses cv_state=CONTINUOUS_FIRE to select
    continuous fire mode rather than gating on alignment time.
    """

    def __init__(self, context: ParticleFilterAutoAimContext):
        super().__init__(
            name="full_state_continuous_fire",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["solution"],
        )
        self._pf: Optional[ParticleFilter] = None

        self._g: float = -(abs(BALLISTIC.gravity) / METERS_TO_CM)    # m/s^2, negative
        self._v: float = BALLISTIC.projectile_velocity / METERS_TO_CM  # m/s
        self._z_offset: float = BALLISTIC.z_offset / METERS_TO_CM      # m
        self._barrel: np.ndarray = np.asarray(BALLISTIC.barrel_offset, dtype=float)  # m

    def set_particle_filter(self, pf: ParticleFilter) -> None:
        """Inject the particle filter instance created by the engine."""
        self._pf = pf

    @real("GPU")
    def _run_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        if estimate is None or target_robot is None:
            print("FullStateContinuousFireModule: missing estimate or target_robot.")
            return None

        if self._pf is None:
            raise RuntimeError("Particle filter not set. Engine must inject it in initialize().")

        # Advance state to firing time (processing delay + feeder actuation lag)
        time_since_estimate = time.perf_counter() - estimate.timestamp
    
        pred = self._pf.prediction(time_since_estimate)

        p_t = np.array([pred[0] / METERS_TO_CM, pred[1] / METERS_TO_CM, self._z_offset])
        v_t = np.array([pred[2] / METERS_TO_CM, pred[3] / METERS_TO_CM, 0.0])
        tht_t = float(pred[4])
        omg_t = float(pred[5])

        a_radius = estimate.a_radius / METERS_TO_CM
        b_radius = estimate.b_radius / METERS_TO_CM

        result = _cascade_solve(
            b=self._barrel,
            s=self._v,
            g=self._g,
            p_t=p_t,
            v_t=v_t,
            tht_t=tht_t,
            omg_t=omg_t,
            a_radius=a_radius,
            b_radius=b_radius,
        )

        if not result['success']:
            print(f"FullStateContinuousFireModule: solver failed (residual={result['residual']:.4f})")
            return None

        return BallisticSolution(
            pitch=result['pitch'],
            yaw=result['yaw'],
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
        )

    @mock
    def _run_mock_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        return BallisticSolution(
            pitch=np.deg2rad(10),
            yaw=np.deg2rad(20),
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
        )
