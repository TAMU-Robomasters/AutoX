"""Process + measurement model for the full-robot EKF (turret frame).

Ported from TAMU-Robomasters/Full-Robot-EKF @5672b45 (``transition_function.py``,
``state_to_panels_corner_2D.py``) and re-expressed in AutoX's canonical turret
frame (see ``src/types/autoaim.py``): x right, y forward, z up, metres.

STATE (10)
    [X, Y, ZA, VX, VY, THETA, OMEGA, LOG_RA, LOG_RB, H]
    X, Y      robot centre (turret frame, m)
    ZA        height of the even panel pair (panels 0 & 2) light-bar centres (m);
              constant-position (random walk) -- no vertical velocity state
              (learner decision 2026-11-11: robots don't move vertically)
    VX, VY    horizontal velocity (m/s)
    THETA     heading, from +x CCW (rad); panel k's outward normal is at
              THETA + k*90deg (AutoX convention -- NOT the EKF repo's numbering)
    OMEGA     yaw rate (rad/s, CCW-positive)
    LOG_RA    log radius of the even pair (panels 0 & 2)
    LOG_RB    log radius of the odd pair (panels 1 & 3)
    H         z_odd - z_even: signed height of the odd pair above the even pair (m)

MEASUREMENT h(x)
    For one panel: its two light bars (left, right as seen in the image) x
    4 corners (TL, TR, BR, BL) x (u, v) pixels = 16 values. Turret-frame points
    are taken into the camera with the inverse of the MCU's camera->turret 4x4
    (which works in pnp.py's relabeled camera axes: right, forward, up), then
    relabeled to OpenCV axes (x right, y down, z forward) and projected with the
    AutoX camera intrinsics + distortion.
"""

from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

X, Y, ZA, VX, VY, THETA, OMEGA, LOG_RA, LOG_RB, H = range(10)
STATE_SIZE = 10

# Panel / light-bar geometry (state_to_panels_corner_2D.py), metres.
PANEL_WIDTH = 0.137
LIGHT_HEIGHT = 0.057
LIGHT_WIDTH = 0.008
PANEL_TILT = np.radians(15.0)  # top of the panel leans in toward the robot centre

MEAS_PER_PANEL = 16  # 2 lights x 4 corners x (u, v)

_HW = PANEL_WIDTH / 2.0
_HH = LIGHT_HEIGHT / 2.0
# (2 lights [left, right], 4 corners [TL, TR, BR, BL], (along-right, along-up))
_LIGHT_LOCAL = np.array(
    [
        [(-_HW, _HH), (-_HW + LIGHT_WIDTH, _HH), (-_HW + LIGHT_WIDTH, -_HH), (-_HW, -_HH)],
        [(_HW - LIGHT_WIDTH, _HH), (_HW, _HH), (_HW, -_HH), (_HW - LIGHT_WIDTH, -_HH)],
    ],
    dtype=np.float64,
)


def wrap_angle(angle):
    """Wrap to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Process model (constant velocity, constant yaw rate)
# ---------------------------------------------------------------------------


def f(x: np.ndarray, dt: float) -> np.ndarray:
    """Propagate the state ``dt`` seconds (height and geometry are constant)."""
    out = np.array(x, dtype=np.float64)
    out[X] += out[VX] * dt
    out[Y] += out[VY] * dt
    out[THETA] = wrap_angle(out[THETA] + out[OMEGA] * dt)
    return out


def f_jacobian(dt: float) -> np.ndarray:
    """dF/dx of :func:`f` (linear, so state-independent)."""
    F = np.eye(STATE_SIZE)
    F[X, VX] = F[Y, VY] = F[THETA, OMEGA] = dt
    return F


# ---------------------------------------------------------------------------
# Panel geometry (turret frame)
# ---------------------------------------------------------------------------


def radii(x: np.ndarray) -> Tuple[float, float]:
    """(r_even, r_odd) in metres."""
    return float(np.exp(x[LOG_RA])), float(np.exp(x[LOG_RB]))


def panel_centers(x: np.ndarray) -> np.ndarray:
    """(4, 3) light-bar-centre point of each panel, turret frame (m)."""
    ra, rb = radii(x)
    k = np.arange(4)
    phi = x[THETA] + k * (np.pi / 2.0)
    r = np.where(k % 2 == 0, ra, rb)
    z = x[ZA] + np.where(k % 2 == 0, 0.0, x[H])
    return np.column_stack([x[X] + r * np.cos(phi), x[Y] + r * np.sin(phi), z])


def light_points(x: np.ndarray, panels: List[int]) -> np.ndarray:
    """(len(panels), 2, 4, 3) light-bar corner points in the turret frame (m).

    Per panel: lights [left, right] as seen from outside the robot, corners
    [TL, TR, BR, BL]. "Right" is the panel tangent that maps to image-right when
    the panel faces the camera (``[-sin, cos]`` of the normal angle); "up" is
    world-up tilted back toward the robot centre by ``PANEL_TILT``.
    """
    centers = panel_centers(x)[panels]                       # (P, 3)
    phi = x[THETA] + np.asarray(panels) * (np.pi / 2.0)      # (P,)
    c, s = np.cos(phi), np.sin(phi)
    zeros = np.zeros_like(c)
    normal = np.stack([c, s, zeros], axis=1)                 # (P, 3)
    right = np.stack([-s, c, zeros], axis=1)
    up = np.cos(PANEL_TILT) * np.array([0.0, 0.0, 1.0]) - np.sin(PANEL_TILT) * normal
    lx = _LIGHT_LOCAL[None, :, :, 0, None]                   # (1, 2, 4, 1)
    ly = _LIGHT_LOCAL[None, :, :, 1, None]
    return (
        centers[:, None, None, :]
        + lx * right[:, None, None, :]
        + ly * up[:, None, None, :]
    )


# ---------------------------------------------------------------------------
# Camera projection
# ---------------------------------------------------------------------------


class Camera:
    """Turret-frame points -> distorted pixels for one frame's extrinsic."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        camera_to_turret: np.ndarray,
    ) -> None:
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        M = np.asarray(camera_to_turret, dtype=np.float64)
        T = np.linalg.inv(M)  # turret -> relabeled camera (right, fwd, up)
        # Fold the relabel to OpenCV axes (x=right, y=-up, z=fwd) into the rotation.
        relabel = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        self._R = relabel @ T[:3, :3]
        self._t = relabel @ T[:3, 3]

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        """(..., 3) turret points -> (..., 3) OpenCV camera points."""
        return pts @ self._R.T + self._t

    def project(self, pts: np.ndarray) -> Optional[np.ndarray]:
        """(..., 3) turret points -> (..., 2) pixels; None if any point is behind the camera."""
        cam = self.to_camera(pts)
        if np.any(cam[..., 2] <= 1e-3):
            return None
        uv, _ = cv.projectPoints(
            cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist
        )
        return uv.reshape(*pts.shape[:-1], 2)


def h(x: np.ndarray, cam: Camera, panels: List[int]) -> Optional[np.ndarray]:
    """Measurement prediction: flat ``(len(panels) * 16,)`` pixel vector, or None."""
    uv = cam.project(light_points(x, panels))
    return None if uv is None else uv.reshape(-1)


def h_jacobian(
    x: np.ndarray, cam: Camera, panels: List[int], eps: float = 1e-5
) -> Optional[np.ndarray]:
    """Central-difference dh/dx, all 2*STATE_SIZE perturbations projected in one call."""
    steps = np.eye(STATE_SIZE) * eps
    states = np.concatenate([x + steps, x - steps])         # (2N, N)
    pts = np.stack([light_points(s, panels) for s in states])
    uv = cam.project(pts)
    if uv is None:
        return None
    uv = uv.reshape(2, STATE_SIZE, -1)
    return ((uv[0] - uv[1]) / (2.0 * eps)).T                # (n_meas, N)


def visible_panels(x: np.ndarray, cam: Camera) -> List[int]:
    """The 3 panels nearest the camera (the one directly behind is culled)."""
    depth = cam.to_camera(panel_centers(x))[:, 2]
    return sorted(np.argsort(depth)[:3].tolist())


def panel_pixel_centers(x: np.ndarray, cam: Camera, panels: List[int]) -> Optional[np.ndarray]:
    """(len(panels), 2) image centre of each panel's 8 light corners."""
    uv = cam.project(light_points(x, panels))
    return None if uv is None else uv.reshape(len(panels), 8, 2).mean(axis=1)
