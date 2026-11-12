"""10-state full-robot EKF (turret frame).

Ported from TAMU-Robomasters/Full-Robot-EKF @5672b45 ``full_robot_ekf.py``.
Kept from the original: initial covariance, physical clamps, 2 px pixel
noise, two-tier innovation gate, Joseph-form update.

Changed in the port:
  * state is in the turret frame (z up) and the camera extrinsic is inside h(x)
    (``model.Camera``), so the filter survives gimbal motion;
  * measurements are whole panels (2 lights x 4 corners) from AutoX's classical
    detector, associated to the nearest predicted visible panel by pixel centre
    (replaces the light-level ``score.pair_lights``);
  * NO velocity decay (learner decision, 2026-11-10). The repo shrank
    VX/VY/VZ/OMEGA on every predict as a workaround for an earlier CV bug; it
    biased tracked velocity/omega low (synthetic: |v| ~1/3 low, omega ~20% low);
  * ``seed()`` initialises from a PnP panel instead of a fixed 2 m guess;
    ``seed_pair()`` from two adjacent panels when both are seen (2026-11-11);
  * process noise redesigned (learner, 2026-11-11), see ``_Q``: continuous
    white-noise acceleration (spectral density q, exact for any dt) for x/y and
    for theta/omega with separate q's, height a random walk (no VZ state).
    Starting values are AutoX's full-state KF numbers; tune on the robot.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.subsystems.estimation.ekf import model
from src.subsystems.estimation.ekf.model import (
    H, LOG_RA, LOG_RB, OMEGA, STATE_SIZE, THETA, VX, VY, X, Y, ZA,
)
from src.subsystems.estimation.robot_constants import measure_adjacent_pair

LOG_R_MIN, LOG_R_MAX = np.log(0.10), np.log(0.30)
H_MIN, H_MAX = -0.10, 0.10
INIT_RADIUS = 0.25        # m; also the seed's inward offset from the panel
INIT_H = 0.0              # m; AutoX seeds from a measured panel, so start level
PAIR_RADIUS_RANGE = (0.05, 0.50)  # m; sanity range for a two-panel seed (as Full-Robot-EKF)


@dataclass
class UpdateResult:
    accepted: bool
    pairs: Dict[int, int] = field(default_factory=dict)  # detection index -> panel k
    innovation: Optional[np.ndarray] = None


class FullRobotEKF:
    """10-state EKF for one robot (all four armor panels)."""

    def __init__(self) -> None:
        self.x = np.zeros(STATE_SIZE)
        self.P = np.eye(STATE_SIZE)
        self.initialized = False

        # Process noise -- TUNING KNOBS. Spectral densities of continuous white
        # noise; sqrt(q * T) is how much the driven quantity can change in T s.
        # Starting values = AutoX full-state KF (motion_models.ConstantVelocity
        # q_vx=q_vy=100 cm/s^1.5, AngleKF q_omega=0.6, HeightKF q=0.5 cm/sqrt(s)).
        # q_xy lowered 1.0 -> 0.1 (learner, 2026-11-11, tuned on the robot in the
        # Full-Robot-EKF repo: velocity stopped "flying away"; 100 ms-ahead
        # prediction 15 px vs 19 px). Over 0.5 s v can now change ~0.22 m/s.
        self._q_xy = 0.1                 # m^2/s^3   (acceleration, x and y)
        self._q_omega = 0.36             # rad^2/s^3 (angular acceleration)
        self._q_z = 0.005 ** 2           # m^2/s     (height random walk)
        self._geom_var = 0.0005 ** 2     # radii/h random walk (from the EKF repo)
        self._pixel_var = 2.0 ** 2              # px^2 per corner coordinate

        # Two-tier innovation gate (max |innovation| per coordinate, px).
        self._gate_init_px = 600.0
        self._gate_track_px = 200.0
        self.tracking_thres = 10
        self.accepted_count = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.__init__()

    def seed(self, panel_pos_m: np.ndarray, panel_yaw: float) -> None:
        """Initialise from one PnP panel (turret frame, metres; yaw = outward normal).

        The panel becomes panel 0 (theta = its normal angle), the centre sits
        ``INIT_RADIUS`` behind it along the inward normal, and z_a is its height.
        """
        n = np.array([np.cos(panel_yaw), np.sin(panel_yaw)])
        self.x = np.zeros(STATE_SIZE)
        self.x[X:Y + 1] = panel_pos_m[:2] - INIT_RADIUS * n
        self.x[ZA] = panel_pos_m[2]
        self.x[THETA] = model.wrap_angle(panel_yaw)
        self.x[LOG_RA] = self.x[LOG_RB] = np.log(INIT_RADIUS)
        self.x[H] = INIT_H
        self.P = np.diag([
            0.3 ** 2, 0.3 ** 2, 0.1 ** 2,          # X, Y, ZA (PnP is decent)
            0.5 ** 2, 0.5 ** 2,                    # VX, VY
            np.deg2rad(10.0) ** 2,                 # THETA
            np.deg2rad(360.0) ** 2,                # OMEGA
            0.005 ** 2, 0.005 ** 2, 0.005 ** 2,    # LOG_RA, LOG_RB, H (as in the repo)
        ])
        self.accepted_count = 0
        self.initialized = True

    def seed_pair(
        self, pos_a: np.ndarray, yaw_a: float, pos_b: np.ndarray, yaw_b: float,
    ) -> bool:
        """Initialise from two adjacent PnP panels (turret frame, metres).

        Panel a becomes panel 0 (theta = its yaw), b is the odd panel next to it.
        Centre, both radii and h come from ``measure_adjacent_pair``
        (c = (p_b . u_a) u_a + (p_a . u_b) u_b). Returns False, leaving the filter
        untouched, if the yaws are not ~90 deg apart or the centre is not behind
        both panels within ``PAIR_RADIUS_RANGE``; the caller then seeds from one.
        """
        pa = np.asarray(pos_a, dtype=np.float64)
        pb = np.asarray(pos_b, dtype=np.float64)
        m = measure_adjacent_pair(pa, yaw_a, pb, yaw_b)
        if m is None:
            return False
        u_a = -np.array([np.cos(yaw_a), np.sin(yaw_a)])   # inward, panel a -> centre
        u_b = -np.array([np.cos(yaw_b), np.sin(yaw_b)])
        ra = float((pb[:2] - pa[:2]) @ u_a)                # signed: > 0 = behind a
        c = pa[:2] + ra * u_a
        rb = float((c - pb[:2]) @ u_b)                     # > 0 = behind b
        lo, hi = PAIR_RADIUS_RANGE
        if not (lo <= ra <= hi and lo <= rb <= hi):
            return False
        self.seed(pa, yaw_a)                               # P0, counters, z_a, theta
        self.x[X:Y + 1] = c
        self.x[LOG_RA] = np.clip(np.log(m.r_a), LOG_R_MIN, LOG_R_MAX)
        self.x[LOG_RB] = np.clip(np.log(m.r_b), LOG_R_MIN, LOG_R_MAX)
        self.x[H] = np.clip(-m.dz, H_MIN, H_MAX)           # dz = z_a - z_b; H = z_odd - z_even
        return True

    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        dt = float(np.clip(dt, 1e-4, 0.5))
        self.x = model.f(self.x, dt)
        F = model.f_jacobian(dt)
        self._clamp()

        self.P = F @ self.P @ F.T + self._Q(dt)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, detections: List[np.ndarray], cam: model.Camera) -> UpdateResult:
        """Update with detected panels, each a (2, 4, 2) light-corner array (px)."""
        if not detections:
            return UpdateResult(False)

        pairs = self.associate(detections, cam)
        if not pairs:
            return UpdateResult(False)

        det_idx = sorted(pairs)
        panels = [pairs[d] for d in det_idx]
        z = np.concatenate([np.asarray(detections[d], dtype=np.float64).reshape(-1) for d in det_idx])
        h_pred = model.h(self.x, cam, panels)
        Hj = model.h_jacobian(self.x, cam, panels)
        if h_pred is None or Hj is None:
            return UpdateResult(False, pairs)

        innovation = z - h_pred
        gate = self._gate_track_px if self.accepted_count >= self.tracking_thres else self._gate_init_px
        if np.max(np.abs(innovation)) > gate:
            return UpdateResult(False, pairs, innovation)

        R = np.eye(len(z)) * self._pixel_var
        S = Hj @ self.P @ Hj.T + R
        try:
            K = np.linalg.solve(S, Hj @ self.P).T
        except np.linalg.LinAlgError:
            return UpdateResult(False, pairs, innovation)

        x_new = self.x + K @ innovation
        if not np.all(np.isfinite(x_new)):
            return UpdateResult(False, pairs, innovation)
        self.x = x_new
        self._clamp()

        I_KH = np.eye(STATE_SIZE) - K @ Hj
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.accepted_count += 1
        return UpdateResult(True, pairs, innovation)

    def associate(self, detections: List[np.ndarray], cam: model.Camera) -> Dict[int, int]:
        """Greedy nearest-pixel-centre match: detection index -> visible panel k."""
        vis = model.visible_panels(self.x, cam)
        pred = model.panel_pixel_centers(self.x, cam, vis)
        if pred is None:
            return {}
        det = np.array([np.asarray(d).reshape(-1, 2).mean(axis=0) for d in detections])
        dist = np.linalg.norm(det[:, None, :] - pred[None, :, :], axis=2)  # (D, V)
        pairs: Dict[int, int] = {}
        for _ in range(min(dist.shape)):
            d, v = np.unravel_index(np.argmin(dist), dist.shape)
            if not np.isfinite(dist[d, v]):
                break
            pairs[int(d)] = vis[int(v)]
            dist[d, :] = np.inf
            dist[:, v] = np.inf
        return pairs

    # ------------------------------------------------------------------

    def _clamp(self) -> None:
        self.x[LOG_RA] = np.clip(self.x[LOG_RA], LOG_R_MIN, LOG_R_MAX)
        self.x[LOG_RB] = np.clip(self.x[LOG_RB], LOG_R_MIN, LOG_R_MAX)
        self.x[H] = np.clip(self.x[H], H_MIN, H_MAX)
        self.x[THETA] = model.wrap_angle(self.x[THETA])

    def _Q(self, dt: float) -> np.ndarray:
        """Continuous white-noise acceleration per (position, rate) pair:
        q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]. Height and geometry: q * dt."""
        Q = np.zeros((STATE_SIZE, STATE_SIZE))
        for pos, vel, q in ((X, VX, self._q_xy), (Y, VY, self._q_xy),
                            (THETA, OMEGA, self._q_omega)):
            Q[pos, pos] = q * dt ** 3 / 3.0
            Q[pos, vel] = Q[vel, pos] = q * dt ** 2 / 2.0
            Q[vel, vel] = q * dt
        Q[ZA, ZA] = self._q_z * dt
        for i in (LOG_RA, LOG_RB, H):
            Q[i, i] = self._geom_var * dt
        return Q
