"""Particle-filter estimation module.

Takes a target robot and produces a ``RobotStateEstimate`` by feeding the
robot's panel observations into a GPU-accelerated particle filter.
"""

import time
from typing import Optional

import cupy as cp
import numpy as np

from src.core.module import Module, real
from src.subsystems.particle_filter import ParticleFilter
from src.types.autoaim import (
    EnemyRobot,
    ParticleFilterAutoAimContext,
    RobotStateEstimate,
)


def _default_particle_filter() -> ParticleFilter:
    """Create a particle filter with default hyperparameters."""
    Q = cp.diag(cp.array([10, 10, 20, 20, 0.1, 1, 0.5], dtype=cp.float32))
    R = cp.diag(cp.array([50, 50,15], dtype=cp.float32))
    prior = cp.array([0, 0, 0, 0, 0, 0, 0], dtype=cp.float32)
    return ParticleFilter(num_particles=30_000, Q=Q, R=R, prior=prior, num_meas=3)


class ParticleFilterEstimationModule(Module[ParticleFilterAutoAimContext]):
    """Estimate robot state using a particle filter."""

    def __init__(self, context: ParticleFilterAutoAimContext):
        super().__init__(
            name="particle_filter_estimation",
            context=context,
            inputs=["target_robot"],
            outputs=["estimate"],
        )
        self._pf: Optional[ParticleFilter] = None
        self._start_loop_time: float = 0.0
        self._initialised: bool = False

    def set_particle_filter(self, pf: ParticleFilter) -> None:
        """Inject the particle filter instance created by the engine."""
        self._pf = pf
        self._initialised = False

    @property
    def pf(self) -> ParticleFilter:
        """Expose the underlying particle filter (used by engine for reinit)."""
        if self._pf is None:
            raise RuntimeError("Particle filter has not been initialized for estimation module")
        return self._pf

    def reset(self) -> None:
        """Mark the filter as uninitialised so it will re-init on next observation."""
        self._initialised = False

    @real()
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RobotStateEstimate]:
        """Run one filter iteration on the target robot's panels."""
        if target_robot is None or not target_robot.panels:
            return None

        panel = target_robot.panels[0]
        if panel.position is None:
            return None

        if self._pf is None:
            raise RuntimeError("Particle filter is not set. Engine must inject it in initialize()")

        # Build prior if this is the first observation (or after reset)
        if not self._initialised:
            default_radius = 21.0
            prior = cp.array(
                [
                    panel.position[0],
                    panel.position[1],
                    0,
                    0,
                    panel.yaw,
                    0,
                    default_radius,
                ],
                dtype=cp.float32,
            )
            self.pf.reinit(prior)
            self._initialised = True
            self._start_loop_time = time.perf_counter()

        current_time = time.perf_counter() - self._start_loop_time
        measurement = cp.array(
            [panel.position[0], panel.position[1], panel.yaw], dtype=cp.float32
        )
        estimate, confidence = self.pf.update(current_time, measurement)

        return RobotStateEstimate(
            value=estimate,
            timestamp=time.perf_counter(),
            confidence=float(confidence),
        )
