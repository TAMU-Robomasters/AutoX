"""Full-state shot-timing ballistic module.

Takes a ``RobotStateEstimate`` and computes the pitch, yaw, and alignment time
needed to hit the target. Intended for fast-spinning targets where the shooter
must wait for a panel to face the correct direction before firing.
"""
from src.core.driver import mock

import math
import time
from typing import Optional, Tuple

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
        print(f"Discriminant is negative, no real solutions for theta (disc={disc:.2f}).")
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


class FullStateShotTimingModule(Module[ParticleFilterAutoAimContext]):
    """Compute a shot-timed firing solution from the robot state estimate.

    For fast-spinning targets: computes when a panel will face the shooter and
    returns the alignment time so the embedded system gates the shot.
    """

    def __init__(self, context: ParticleFilterAutoAimContext):
        super().__init__(
            name="full_state_shot_timing",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["solution"],
        )
        self._pf: Optional[ParticleFilter] = None

        self._g: float = BALLISTIC.gravity
        self._v: float = BALLISTIC.projectile_velocity
        self._z_offset: float = BALLISTIC.z_offset

    def set_particle_filter(self, pf: ParticleFilter) -> None:
        """Inject the particle filter instance created by the engine."""
        self._pf = pf

    @real("GPU")
    def _run_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Compute pitch, yaw, and alignment time from the estimate."""
        if estimate is None or target_robot is None:
            print("FullStateShotTimingModule: missing estimate or target_robot.")
            return None
        est = estimate.value
        d = math.sqrt(float(est[0]) ** 2 + float(est[1]) ** 2) - float(23.5)
        if d <= 0:
            print(f"FullStateShotTimingModule: target too close (d={d:.2f}cm).")
            return None

        solutions = _theta_solver(d, self._z_offset, self._g, self._v)
        if solutions is None:
            print(f"WARNING: Theta solver failed for d={d:.2f}cm, delta_z={self._z_offset:.2f}cm")
            return None

        pitch, t = min(solutions, key=lambda s: s[1])

        time_since_estimate = time.perf_counter() - estimate.timestamp
        feeder_delay = 0.15

        if self._pf is None:
            raise RuntimeError("Particle filter not set. Engine must inject it in initialize().")
        prediction = self._pf.prediction(t + time_since_estimate + feeder_delay)

        yaw = float(
            
            np.arctan2(float(prediction[1]), float(prediction[0]))
        ) - np.deg2rad(90)

        theta0 = float(prediction[4])
        omega = float(prediction[5])

        thetas = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]) + theta0

        if omega > 0:
            angle_to_travel = (yaw - thetas) % (2 * np.pi)
            alignment_time = float(np.min(angle_to_travel) / omega)
        elif omega < 0:
            angle_to_travel = (thetas - yaw) % (2 * np.pi)
            alignment_time = float(np.min(angle_to_travel) / (-omega))
        else:
            alignment_time = 0.0

        alignment_time_ms = int(alignment_time * 1000)

        return BallisticSolution(
            pitch=float(pitch),
            yaw=float(yaw),
            alignment_time_ms=alignment_time_ms,
        )

    @mock
    def _run_mock_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        return BallisticSolution(
            pitch=np.deg2rad(10),
            yaw=np.deg2rad(20),
            alignment_time_ms=500,
        )
