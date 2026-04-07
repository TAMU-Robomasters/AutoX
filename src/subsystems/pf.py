"""Particle-filter estimation module.

Takes a target robot and produces a ``RobotStateEstimate`` by feeding the
robot's panel observations into a GPU-accelerated particle filter.
"""
import time
from typing import Optional

import cupy as cp
import numpy as np

from src.core.module import Module, real, mock
from src.subsystems.particle_filter import ParticleFilter
from src.types.autoaim import (
    EnemyRobot,
    ParticleFilterAutoAimContext,
    RobotStateEstimate, ArmorPanel,
)


def _default_particle_filter() -> ParticleFilter:
    """Create a particle filter with default hyperparameters."""
    Q = cp.diag(cp.array([10 ** 2, 10 ** 2, 20 ** 2, 20 ** 2, 0.01 ** 2, 0.5 ** 2], dtype=cp.float32))
    R = cp.diag(cp.array([10 ** 2, 10 ** 2, 0.2 ** 2], dtype=cp.float32))
    prior = cp.array([0, 0, 0, 0, 0, 0], dtype=cp.float32)
    return ParticleFilter(num_particles=35_000, Q=Q, R=R, prior=prior, num_meas=3)


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
        self._initialised: bool = False
        self.is_their_target_prev = False
        self.is_their_target = False
        self.last_update_time = time.monotonic()

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

    @real("GPU")
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RobotStateEstimate]:
        """Run one filter iteration on the target robot's panels."""
        if target_robot is None:
            self.is_their_target_prev = False
            return None
        
        self.is_their_target = True
        if self._pf is None:
            raise RuntimeError("Particle filter is not set. Engine must inject it in initialize()")

        panel: ArmorPanel = target_robot.panels[0]
        # Build prior if this is the first observation (or after reset)
        if self.is_their_target and not self.is_their_target_prev: # new target
            default_radius = 21.0
            prior = cp.array(
                [
                    panel.position[0],
                    panel.position[1],
                    0,
                    0,
                    panel.yaw,
                    0
                ],
                dtype=cp.float32,
            )
            last_update_time = self.last_update_time
            self.pf.reinit(prior)

        self.is_their_target_prev = self.is_their_target
        current_time = time.perf_counter() - self.ctx.start_loop_time if not self.ctx.start_loop_time is None else 0.0
        if self.ctx.new_observation:
            measurement = cp.array(
                [panel.position[0], panel.position[1], panel.yaw], dtype=cp.float32
            )
            dt = self.ctx.frame_ts - self.last_update_time
            self.last_update_time = self.ctx.frame_ts
            estimate, confidence = self.pf.update(dt, measurement)
        else:
            current_time = time.monotonic()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            estimate, confidence = self.pf.update_with_no_observation(dt)

        return RobotStateEstimate(
            value=estimate,
            timestamp=time.perf_counter(),
            confidence=float(confidence),
        )

    @mock
    def _run_mock_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RobotStateEstimate]:
        """Mock estimation that returns a fixed state estimate."""
        if target_robot is None or not target_robot.panels:
            return None

        panel = target_robot.panels[0]
        if panel.position is None:
            return None

        # Return a dummy estimate with the panel's position and yaw, zero velocity,
        # and a fixed radius. Confidence is set to 1.0 for simplicity.
        return RobotStateEstimate(
            value=np.array(
                [
                    panel.position[0],
                    panel.position[1],
                    0,
                    0,
                    panel.yaw,
                    0,
                    21.0,
                ],
                dtype=np.float32,
            ),
            timestamp=time.perf_counter(),
            confidence=1.0,
        )

