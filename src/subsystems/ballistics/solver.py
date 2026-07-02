"""Canonical ballistic solver geometry, shared by all ballistic modules.

This file is the single source of truth for the coordinate convention; the
panel-offset/normal math here MUST stay consistent with the estimator's
back-projection (``PositionKF.back_project``) and ``PanelTrackingModule``.

CANONICAL CONVENTION (turret/ballistic frame)
    x -> right, y -> forward (out of the barrel at yaw 0), z -> up.
    All *internal* angles (robot heading ``theta``, panel yaw) are measured
    from +x, counter-clockwise -- standard ``atan2(y, x)``.
    Panel ``k`` of a robot at heading ``theta`` sits at
        ``center + r_k * [cos(theta + k*pi/2), sin(theta + k*pi/2)]``
    and that angle is also the panel's outward-normal direction.

MCU YAW (output boundary only)
    The embedded yaw convention has 0 along +y: ``yaw_mcu = atan2(y, x) - pi/2``
    (:func:`mcu_yaw_from_xy`). NOTE the deliberate asymmetry inside the
    projectile residuals: the solver's ``yaw`` unknown is *natively* in the MCU
    convention (yaw=0 fires along +y -- the ``-s*t*sin(yaw)`` / ``+s*t*cos(yaw)``
    terms), while the target-side ``theta`` terms use the internal convention.
    Both are correct; do not "fix" one to match the other.

Units: any length unit works as long as positions, speeds, gravity, and barrel
offsets agree (the full-state continuous-fire module uses metres).
"""

from typing import Callable, Dict

import numpy as np
from scipy.optimize import root


def mcu_yaw_from_xy(x: float, y: float) -> float:
    """Convert a target direction to the MCU yaw convention (0 at +y), wrapped to [-pi, pi]."""
    yaw = np.arctan2(y, x) - np.pi / 2.0
    return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


def panel_normals(theta: float) -> np.ndarray:
    """Outward unit normals of the four panels, rows ``k = 0..3``.

    Row ``k`` is ``[cos(theta + k*pi/2), sin(theta + k*pi/2)]`` -- the same
    angle convention as the estimator's back-projection (theta from +x, CCW).
    """
    angles = theta + np.arange(4) * (np.pi / 2.0)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def select_panel(p_t: np.ndarray, tht_t: float) -> int:
    """Index of the panel whose outward normal best faces the turret (at the origin)."""
    facing = -p_t[:2]
    norm = np.linalg.norm(facing)
    if norm < 1e-9:
        return 0
    return int(np.argmax(panel_normals(tht_t) @ (facing / norm)))


def select_panel_at_time(
    p_t: np.ndarray, v_t: np.ndarray, tht_t: float, omg_t: float, t: float
) -> int:
    """Like :func:`select_panel`, but for the robot's pose extrapolated ``t`` seconds ahead."""
    p_future = p_t + v_t * t
    tht_future = tht_t + omg_t * t
    return select_panel(p_future, tht_future)


def init_guess(p_t: np.ndarray, speed: float, gravity: float) -> np.ndarray:
    """Closed-form [yaw, pitch, time] starting point for the LM solvers.

    ``p_t`` is the target position [x, y, z]; the yaw component is already in
    the MCU convention (matches the projectile residuals).
    """
    g = abs(gravity)
    x = np.hypot(p_t[0], p_t[1])  # horizontal range
    y = p_t[2]                    # vertical offset

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

    return np.array([mcu_yaw_from_xy(p_t[0], p_t[1]), pitch0, t0])


def make_f_no_spin(
    b: np.ndarray,
    s: float,
    g: float,
    p_t: np.ndarray,
    v_t: np.ndarray,
    a_t: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Residual function for hitting a (optionally accelerating) point target.

    The projectile leaves the barrel tip ``b`` (rotated by yaw/pitch) at speed
    ``s``, drops under gravity ``g`` (negative), and must meet the target at
    flight time ``t``. With ``a_t=None`` the target moves at constant velocity
    (``p_t + v_t * t``); with a ``[ax, ay, az]`` it moves at constant
    acceleration (``p_t + v_t * t + 0.5 * a_t * t^2``) -- the constant-velocity
    case is recovered exactly when ``a_t`` is zero. Unknowns ``x = [yaw, pitch,
    t]`` with yaw in the MCU convention (yaw=0 fires along +y).
    """
    a = np.zeros(3) if a_t is None else np.asarray(a_t, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        yaw, pitch, t = x
        return np.array([
            b[0] * np.cos(yaw) - np.sin(yaw) * (b[1] * np.cos(pitch) - b[2] * np.sin(pitch))
                - s * t * np.sin(yaw) * np.cos(pitch) - p_t[0] - v_t[0] * t - 0.5 * a[0] * t**2,
            b[0] * np.sin(yaw) + np.cos(yaw) * (b[1] * np.cos(pitch) - b[2] * np.sin(pitch))
                + s * t * np.cos(yaw) * np.cos(pitch) - p_t[1] - v_t[1] * t - 0.5 * a[1] * t**2,
            b[1] * np.sin(pitch) + b[2] * np.cos(pitch) + s * t * np.sin(pitch)
                + 0.5 * g * t**2 - p_t[2] - v_t[2] * t - 0.5 * a[2] * t**2,
        ])

    return f


def solve_no_spin(
    b: np.ndarray,
    s: float,
    g: float,
    p_t: np.ndarray,
    v_t: np.ndarray,
    tol: float = 1e-4,
    a_t: np.ndarray | None = None,
) -> Dict:
    """LM-solve :func:`make_f_no_spin` for [yaw, pitch, time].

    ``a_t`` (optional ``[ax, ay, az]`` target acceleration) is folded into the
    in-flight projectile arc. Returns a dict with ``success`` (residual < tol),
    ``yaw`` (MCU convention), ``pitch``, ``time``, and ``residual``. On failure
    yaw/pitch/time hold the solver's last iterate (callers must check ``success``).
    """
    f = make_f_no_spin(b, s, g, p_t, v_t, a_t)
    result = root(f, init_guess(p_t, s, g), method="lm", tol=tol)
    residual = float(np.linalg.norm(f(result.x)))
    return {
        "success": residual < tol and 0.0 < result.x[2] < np.inf,
        "yaw": float(result.x[0]),
        "pitch": float(result.x[1]),
        "time": float(result.x[2]),
        "residual": residual,
    }
