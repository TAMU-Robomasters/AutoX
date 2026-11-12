"""Continuous-fire ballistics for the EKF engine.

Ported from TAMU-Robomasters/Full-Robot-EKF @5672b45 ``firing_solutions.solve``
and adapted to AutoX's conventions:
  * turret frame, z up, so gravity drop is compensated along +z;
  * panel k at ``theta + k*90deg`` with even/odd radii and the odd pair raised
    by ``h`` (same geometry as ``estimation/ekf/model.py``);
  * yaw returned in the MCU convention (0 at +y, ``solver.mcu_yaw_from_xy``),
    pitch positive up;
  * no barrel offset (learner decision, 2026-11-09).

Algorithm (unchanged): for each panel k, find the smallest flight time t where
a straight-line shot of speed ``v`` plus gravity drop reaches the panel's
predicted position at t (grid search + bisection). Keep only panels that are the
most camera-facing panel at their own impact time; among those pick the one
closest to the current aim yaw (or the most head-on without one).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.ballistics.solver import mcu_yaw_from_xy
from src.subsystems.estimation.ekf.model import (
    H, LOG_RA, LOG_RB, OMEGA, THETA, VX, VY, X, Y, ZA,
)
from src.toolbox.globals import config
from src.types.autoaim import BallisticSolution, EkfStateEstimate, EnemyRobot

METERS_TO_CM = 100.0
_CONT_FIRE_ALIGNMENT_TIME_MS = 255


@dataclass
class EkfShot:
    panel: int
    t_flight: float
    yaw: float    # MCU convention
    pitch: float  # rad, positive up
    score: float  # facing dot product (more negative = more head-on)


def _panel_pos(k: int, t: float, s: np.ndarray, ra: float, rb: float):
    phi = s[THETA] + s[OMEGA] * t + k * (math.pi / 2.0)
    r = ra if k % 2 == 0 else rb
    return (
        s[X] + s[VX] * t + r * math.cos(phi),
        s[Y] + s[VY] * t + r * math.sin(phi),
        s[ZA] + (s[H] if k % 2 else 0.0),
    )


def _facing_panel(t: float, s: np.ndarray, ra: float, rb: float):
    """Panel whose centre->panel offset points most toward the turret (xy only)."""
    cx, cy = s[X] + s[VX] * t, s[Y] + s[VY] * t
    best_k, best_d = 0, math.inf
    for k in range(4):
        phi = s[THETA] + s[OMEGA] * t + k * (math.pi / 2.0)
        r = ra if k % 2 == 0 else rb
        d = r * (cx * math.cos(phi) + cy * math.sin(phi))
        if d < best_d:
            best_k, best_d = k, d
    return best_k, best_d


def _time_of_flight(k, s, ra, rb, speed, g, t_max, steps=600) -> Optional[float]:
    """Smallest t with |P_k(t) + 0.5 g t^2 z_hat| = speed * t."""
    def fn(t):
        px, py, pz = _panel_pos(k, t, s, ra, rb)
        pz += 0.5 * g * t * t
        return px * px + py * py + pz * pz - (speed * t) ** 2

    if fn(0.0) <= 0:
        return 0.0
    dt = t_max / steps
    t0 = 0.0
    for i in range(1, steps + 1):
        t1 = i * dt
        if fn(t1) <= 0:
            lo, hi = t0, t1
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if fn(mid) > 0 else (lo, mid)
            return 0.5 * (lo + hi)
        t0 = t1
    return None


def solve(
    s: np.ndarray,
    speed: float,
    g: float = 9.81,
    t_max: float = 1.5,
    current_yaw: Optional[float] = None,
) -> Optional[EkfShot]:
    """Firing solution for EKF state ``s`` (metres, already advanced to fire time).

    ``g`` is the positive gravity magnitude. ``current_yaw`` (MCU convention)
    adds hysteresis between equally valid panels. None = no panel faces us at
    its own impact time: hold fire.
    """
    ra, rb = math.exp(s[LOG_RA]), math.exp(s[LOG_RB])
    consistent = []
    for k in range(4):
        t = _time_of_flight(k, s, ra, rb, speed, g, t_max)
        if t is None or t <= 0:
            continue
        winner, dot = _facing_panel(t, s, ra, rb)
        if winner != k:
            continue
        px, py, pz = _panel_pos(k, t, s, ra, rb)
        dz = pz + 0.5 * g * t * t  # aim above the panel by the gravity drop
        n = math.sqrt(px * px + py * py + dz * dz)
        consistent.append(EkfShot(
            panel=k,
            t_flight=t,
            yaw=mcu_yaw_from_xy(px, py),
            pitch=math.asin(dz / n),
            score=dot,
        ))
    if not consistent:
        return None
    if current_yaw is None:
        return min(consistent, key=lambda c: c.score)
    return min(
        consistent,
        key=lambda c: abs((c.yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi),
    )


def advance(s: np.ndarray, dt: float) -> np.ndarray:
    """Constant-velocity / constant-yaw-rate extrapolation (no decay, no clamps)."""
    out = np.array(s, dtype=np.float64)
    out[X] += out[VX] * dt
    out[Y] += out[VY] * dt
    out[THETA] += out[OMEGA] * dt
    return out


class EkfContinuousFireModule(Module):
    """Continuous-fire solution from the EKF state (ported Full-Robot-EKF solver)."""

    def __init__(self, context) -> None:
        super().__init__(
            name="ekf_continuous_fire",
            context=context,
            inputs=["ekf_estimate", "target_robot"],
            outputs=["solution"],
        )
        b = config.ballistic
        self._speed = float(b.projectile_velocity) / METERS_TO_CM  # m/s
        self._g = abs(float(b.gravity)) / METERS_TO_CM             # m/s^2
        self._t_max = float(b.max_time_of_flight)
        self._lookahead = float(b.lookahead_time)
        self._max_range_m = float(b.max_range) / METERS_TO_CM
        #: Last commanded yaw (MCU convention, before yaw_offset); set by the engine.
        self.current_yaw: Optional[float] = None

    @real()
    def _run_solve(
        self, ekf_estimate: Optional[EkfStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        if ekf_estimate is None or target_robot is None:
            return None
        lead = time.perf_counter() - ekf_estimate.timestamp + self._lookahead
        s = advance(ekf_estimate.x, lead)

        if math.hypot(s[X], s[Y]) <= self._max_range_m:
            shot = solve(s, self._speed, self._g, self._t_max, self.current_yaw)
            if shot is not None:
                return BallisticSolution(
                    pitch=shot.pitch,
                    yaw=shot.yaw,
                    alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
                )
        # Out of range / nothing facing us: straight-line aim at the centre, hold fire.
        aim_z = s[ZA] + 0.5 * s[H]
        return BallisticSolution(
            pitch=float(math.atan2(aim_z, math.hypot(s[X], s[Y]))),
            yaw=mcu_yaw_from_xy(float(s[X]), float(s[Y])),
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
            is_confident=False,
        )

    @mock
    def _run_mock_solve(self, ekf_estimate, target_robot) -> Optional[BallisticSolution]:
        return BallisticSolution(
            pitch=np.deg2rad(10), yaw=np.deg2rad(20),
            alignment_time_ms=_CONT_FIRE_ALIGNMENT_TIME_MS,
        )
