"""Ballistic solver module.

Takes a ``RobotStateEstimate`` and computes the pitch, yaw, and alignment time
needed to hit the target.
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


class BallisticSolverModule(Module[ParticleFilterAutoAimContext]):
    """Compute a firing solution from the robot state estimate."""

    def __init__(self, context: ParticleFilterAutoAimContext):
        super().__init__(
            name="ballistic_solver",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["solution"],
        )
        self._pf: Optional[ParticleFilter] = None

        # Load ballistic parameters from config (set in info.yaml)
        self._g: float = BALLISTIC.gravity
        self._v: float = BALLISTIC.projectile_velocity
        self._z_offset: float = BALLISTIC.z_offset

    def set_particle_filter(self, pf: ParticleFilter) -> None:
        """Inject the particle filter instance created by the engine."""
        self._pf = pf

    #TODO: decouple ballistic solver from particle filter prediction by passing predicted state as input instead of estimate + target_robot
    @real("GPU")
    def _run_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Compute pitch, yaw, and alignment time from the estimate."""
        if estimate is None or target_robot is None:
            print("Ballistic solver missing estimate or target_robot.")
            return None
        est = estimate.value
        # Horizontal distance to robot centre minus orbit radius
        d = math.sqrt(float(est[0]) ** 2 + float(est[1]) ** 2) - float(23.5)
        # print(f"Computed horizontal distance to target: {d:.2f}cm (after orbit compensation).")
        if d <= 0:
            print(f"Target is too close for ballistic solution (d={d:.2f}cm).")
            return None

        solutions = _theta_solver(d, target_robot.panels[0].position[2] + self._z_offset, self._g, self._v)
        if solutions is None:
            print(f"WARNING: Theta solver failed to find a solution for d={d:.2f}cm, delta_z={target_robot.panels[0].position[2] + self._z_offset:.2f}cm")
            return None

        # Pick the flatter trajectory (shorter time of flight)
        pitch, t = min(solutions, key=lambda s: s[1])

        # Compensate for processing delay
        time_since_estimate = time.perf_counter() - estimate.timestamp
        feeder_delay = 0.065 # 65 ms

        if self._pf is None:
            raise RuntimeError("Particle filter is not set. Engine must inject it in initialize()")
        prediction = self._pf.prediction(t + time_since_estimate + feeder_delay)

        yaw = float(
            np.arctan2(float(prediction[1]), float(prediction[0]))
        ) - np.deg2rad(90) 

        # Alignment time: when spinning panel will face the shooter
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
        """Mock ballistic solver that returns a fixed solution."""
        return BallisticSolution(
            pitch=np.deg2rad(10),
            yaw=np.deg2rad(20),
            alignment_time_ms=500,
        )
