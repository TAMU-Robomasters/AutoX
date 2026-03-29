"""Ballistic solver module.

Takes a ``RobotStateEstimate`` and computes the pitch, yaw, and alignment time
needed to hit the target.
"""

import math
import time
from typing import Optional, Tuple

import cupy as cp
import numpy as np

from src.core.module import Module, real
from src.subsystems.particle_filter import ParticleFilter
from src.toolbox.globals import config
from src.types.autoaim import (
    BallisticSolution,
    ParticleFilterAutoAimContext,
    RobotStateEstimate,
    EnemyRobot,
)
from src.toolbox.globals import config


BALLISTIC = config.ballistic


def _theta_solver(
    d: float, delta_z: float, g: float, v: float
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Solve the projectile ballistic equation for pitch angle.

    Args:
        d: Horizontal distance to target (cm).
        delta_z: Vertical offset to target (cm, negative = below).
        g: Gravitational acceleration (cm/s^2).
        v: Projectile velocity (cm/s).

    Returns:
        Two (theta, time_of_flight) solutions, or None if no real solution.
    """
    v2 = v * v
    d2 = d * d
    A = (g * d2) / (2.0 * v2)

    disc = d2 - 4.0 * A * (delta_z + A)
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    inv_denom = 1.0 / (2.0 * A)

    tan1 = (d + sqrt_disc) * inv_denom
    tan2 = (d - sqrt_disc) * inv_denom

    theta1 = math.atan(tan1)
    theta2 = math.atan(tan2)

    t1 = d / (v * math.cos(theta1))
    t2 = d / (v * math.cos(theta2))

    return (theta1, t1), (theta2, t2)


class BallisticSolverModule(Module[ParticleFilterAutoAimContext]):
    """Compute a firing solution from the robot state estimate."""

    def __init__(self, context: ParticleFilterAutoAimContext, pf: ParticleFilter):
        super().__init__(
            name="ballistic_solver",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["solution"],
        )
        self._pf = pf

        # Load ballistic parameters from config (set in info.yaml)
        self._g: float = BALLISTIC.gravity
        self._v: float = BALLISTIC.projectile_velocity
        self._z_offset: float = BALLISTIC.z_offset

    @real()
    def _run_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Compute pitch, yaw, and alignment time from the estimate."""
        if estimate is None or target_robot is None:
            return None
        est = estimate.value
        # Horizontal distance to robot centre minus orbit radius
        d = math.sqrt(float(est[0]) ** 2 + float(est[1]) ** 2) - float(est[6])
        if d <= 0:
            return None

        solutions = _theta_solver(d, target_robot.panels[0].position[2] + self._z_offset, self._g, self._v)
        if solutions is None:
            return None

        # Pick the flatter trajectory (shorter time of flight)
        theta, t = min(solutions, key=lambda s: s[1])

        # Compensate for processing delay
        time_offset = time.perf_counter() - estimate.timestamp
        prediction = self._pf.prediction(t + time_offset)

        yaw = float(
            np.arctan2(float(prediction[1]), float(prediction[0]))
        ) - np.deg2rad(88) #?! why is there a magic 88 degree offset?

        # Alignment time: when spinning panel will face the shooter
        theta0 = float(prediction[4])
        omega = float(prediction[5])
        delta = np.arctan2(np.sin(yaw - theta0), np.cos(yaw - theta0))

        if omega > 0:
            alignment_time = (delta % (2 * np.pi)) / omega
        elif omega < 0:
            alignment_time = (delta % (-2 * np.pi)) / omega
        else:
            alignment_time = 0.0

        alignment_time_ms = int(alignment_time * 1000)

        return BallisticSolution(
            pitch=float(theta),
            yaw=float(yaw),
            alignment_time_ms=alignment_time_ms,
        )
