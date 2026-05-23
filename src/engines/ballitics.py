"""
Ballistics solver — simplified, standalone, no acceleration terms.

Solves for [yaw, pitch, travel_time] to hit a spinning robot's armor panel
using a cascade scipy LM approach:
  1. Solve f_no_spin (no panel offset) to estimate travel time.
  2. Forward-predict target position + heading at that time → select panel.
  3. Solve f_spin for that panel, warm-started from the no-spin solution.

Coordinate frame (right-handed, same convention throughout):
  x  →  right
  y  →  forward / out (depth)
  z  →  up

Validated operating envelope (warnings issued outside these bounds):
  projectile speed  :  9 – 12 m/s     (cascade success ≥ 98%)
  target spin       :  |ω| ≤ 5 rad/s
  target speed      :  0 – 4 m/s, horizontal only (ground robots)
  target distance   :  1 m – (v²/g − 4 m)

Acceleration terms (linear a_t, angular alp_t) are omitted.
If your targets have significant acceleration, use ballistics_scipy.py instead.
"""

import warnings
import numpy as np
from scipy.optimize import root


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warn_envelope(
    projectile_speed: float,
    p_t: np.ndarray,
    v_t: np.ndarray,
    omg_t: float,
    gravity: float,
) -> None:
    g = abs(gravity)

    if projectile_speed < 9.0:
        warnings.warn(
            f"projectile_speed={projectile_speed:.2f} m/s is below the tested range "
            f"(9–12 m/s). Cascade success rate at low speeds is untested.",
            UserWarning, stacklevel=3,
        )

    if abs(omg_t) > 5.0:
        warnings.warn(
            f"|omg_t|={abs(omg_t):.2f} rad/s exceeds the tested spin envelope "
            f"(≤5 rad/s). Panel selection accuracy may degrade.",
            UserWarning, stacklevel=3,
        )

    horiz_speed = float(np.hypot(v_t[0], v_t[1]))
    if projectile_speed <= 12.0:
        _speed_limit = 4.0
    elif projectile_speed <= 16.0:
        _speed_limit = 2.0
    elif projectile_speed <= 20.0:
        _speed_limit = 6.0
    elif projectile_speed <= 25.0:
        _speed_limit = 12.0
    else:
        _speed_limit = None   # 25+ m/s: no target-speed constraint found in testing

    if _speed_limit is not None and horiz_speed > _speed_limit:
        warnings.warn(
            f"Horizontal target speed={horiz_speed:.2f} m/s exceeds the tested safe "
            f"limit of {_speed_limit:.0f} m/s for projectile_speed={projectile_speed:.1f} m/s "
            f"(success rate drops below 95% above this threshold).",
            UserWarning, stacklevel=3,
        )

    if abs(v_t[2]) > 1e-3:
        warnings.warn(
            f"v_t[2]={v_t[2]:.3f} m/s — non-zero vertical target velocity is untested. "
            f"Script was validated for ground robots with horizontal-only motion.",
            UserWarning, stacklevel=3,
        )

    horiz_dist = float(np.hypot(p_t[0], p_t[1]))
    max_range  = projectile_speed**2 / g - 4.0
    if horiz_dist > max_range:
        warnings.warn(
            f"Horizontal target distance={horiz_dist:.2f} m exceeds effective range "
            f"{max_range:.2f} m (v²/g − 4 m) for speed={projectile_speed:.1f} m/s. "
            f"Solver may fail.",
            UserWarning, stacklevel=3,
        )


def _init_guess(p_t: np.ndarray, speed: float, gravity: float) -> np.ndarray:
    """Wikipedia low-trajectory formula for pitch; barrel offset ignored."""
    g = abs(gravity)
    x = np.hypot(p_t[0], p_t[1])
    y = p_t[2]

    if x < 1e-9:
        t0 = abs(y) / speed if abs(y) > 1e-9 else 0.1
        return np.array([0.0, np.sign(y) * np.pi / 2, t0])

    discriminant = speed**4 - g * (g * x**2 + 2.0 * y * speed**2)
    pitch0 = np.arctan((speed**2 - np.sqrt(discriminant)) / (g * x)) if discriminant >= 0 else np.pi / 4

    cos_p = np.cos(pitch0)
    t0    = x / (speed * cos_p) if abs(cos_p) > 1e-6 else np.linalg.norm(p_t) / speed

    yaw0 = np.arccos(np.clip(p_t[1] / x, -1.0, 1.0))
    if p_t[0] > 0:
        yaw0 = -yaw0

    return np.array([yaw0, pitch0, t0])


def _panel_normals(tht: float) -> np.ndarray:
    """2-D outward unit normals for all 4 panels at heading tht."""
    return np.array([
        [np.sin(tht),              np.cos(tht)],
        [np.sin(tht + np.pi/2),   np.cos(tht + np.pi/2)],
        [np.sin(tht + np.pi),     np.cos(tht + np.pi)],
        [np.sin(tht - np.pi/2),   np.cos(tht - np.pi/2)],
    ])


def _select_panel(p_t: np.ndarray, tht_t: float) -> int:
    """Pick the armor panel most facing the shooter at current target state."""
    p_xy = -p_t[:2]
    norm = np.linalg.norm(p_xy)
    if norm < 1e-9:
        return 0
    return int(np.argmax(_panel_normals(tht_t) @ (p_xy / norm)))


def _select_panel_at_time(
    p_t: np.ndarray,
    v_t: np.ndarray,
    tht_t: float,
    omg_t: float,
    time: float,
) -> int:
    """Pick the armor panel most facing the shooter at the predicted impact time."""
    p_future   = p_t + v_t * time
    tht_future = tht_t + omg_t * time

    p_xy = -p_future[:2]
    norm = np.linalg.norm(p_xy)
    if norm < 1e-9:
        return 0
    return int(np.argmax(_panel_normals(tht_future) @ (p_xy / norm)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(
    b: list,
    projectile_speed: float,
    gravity: float,
    p_t: list,
    v_t: list,
    tht_t: float,
    omg_t: float,
    a_radius: float,
    b_radius: float,
    init_guess: list = None,
    tol: float = 1e-4,
    check_envelope: bool = True,
) -> dict:
    """
    Cascade ballistics solver. No acceleration terms.

    Parameters
    ----------
    b               : barrel tip offset from ballistic origin [bx, by, bz]
    projectile_speed: muzzle speed (m/s)
    gravity         : gravitational acceleration (negative, e.g. -9.8)
    p_t             : target position [x, y, z]
    v_t             : target linear velocity [x, y, z]
    tht_t           : target yaw angle (radians)
    omg_t           : target angular velocity (rad/s, yaw axis)
    a_radius        : armor panel radius for panels 0 and 2 (forward/back)
    b_radius        : armor panel radius for panels 1 and 3 (right/left)
    init_guess      : warm-start [yaw, pitch, travel_time]; auto-computed if None
    tol             : residual tolerance for convergence
    check_envelope  : emit UserWarnings when inputs are outside tested bounds

    Returns
    -------
    {
        'success' : bool,
        'yaw'     : float (radians),   None on failure
        'pitch'   : float (radians),   None on failure
        'time'    : float (seconds),   None on failure
        'panel'   : int   (0–3),
        'residual': float,
    }

    Armor panel layout (target yaw = 0):
        panel 0 → forward (+y),  radius = a_radius
        panel 1 → right   (+x),  radius = b_radius
        panel 2 → back    (−y),  radius = a_radius
        panel 3 → left    (−x),  radius = b_radius
    """
    b   = np.asarray(b,   dtype=float)
    p_t = np.asarray(p_t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)
    s   = float(projectile_speed)
    g   = float(gravity)

    if check_envelope:
        _warn_envelope(s, p_t, v_t, omg_t, g)

    x0 = _init_guess(p_t, s, g) if init_guess is None else np.asarray(init_guess, dtype=float)

    # --- Step 1: f_no_spin ---
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

    ns_res      = root(f_no_spin, x0, method='lm', tol=tol)
    ns_residual = float(np.linalg.norm(f_no_spin(ns_res.x)))
    ns_ok       = ns_residual < tol

    # --- Step 2: panel selection ---
    if ns_ok:
        panel     = _select_panel_at_time(p_t, v_t, tht_t, omg_t, float(ns_res.x[2]))
        spin_init = ns_res.x
    else:
        panel     = _select_panel(p_t, tht_t)
        spin_init = x0

    # --- Step 3: f_spin for selected panel ---
    r          = a_radius if panel in (0, 2) else b_radius
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

    result   = root(f_spin, spin_init, method='lm', tol=tol)
    residual = float(np.linalg.norm(f_spin(result.x)))
    success  = residual < tol

    if success:
        return {
            'success':  True,
            'yaw':      float(result.x[0]),
            'pitch':    float(result.x[1]),
            'time':     float(result.x[2]),
            'panel':    panel,
            'residual': residual,
        }
    return {
        'success':  False,
        'yaw':      None,
        'pitch':    None,
        'time':     None,
        'panel':    panel,
        'residual': residual,
    }