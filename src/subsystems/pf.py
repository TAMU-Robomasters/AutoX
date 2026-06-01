"""Particle-filter estimation module.

Takes a target robot and produces a ``RobotStateEstimate`` by feeding the
robot's panel observations into a GPU-accelerated particle filter.
"""
import time
from typing import Optional

import numpy as np

from src.core.module import Module, real, mock
from src.subsystems.particle_filter import ParticleFilter
from src.types.autoaim import (
    EnemyRobot,
    ParticleFilterAutoAimContext,
    RobotStateEstimate, ArmorPanel,
)


def _default_particle_filter() -> ParticleFilter:
    """Create a particle filter with default hyperparameters.

    Values ported from full-state-estimation-sim/config.py, scaled from metres
    to centimetres for AutoX's coordinate system (1 m = 100 cm).
    """
    # From sim: [0.7, 0.7, 2.0] m/s, rad/s  →  cm/s, rad/s
    Q_vel    = np.array([70.0, 70.0, 0.3], dtype=np.float32)
    # From sim: [1.0, 1.0, 0.1, 0.1, 0.3, 15.0]  (m, m, m/s, m/s, rad, rad/s)
    init_std = np.array([100.0, 100.0, 10.0, 10.0, 0.3, 15.0], dtype=np.float32)
    prior    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return ParticleFilter(
        num_particles=40_000,
        Q_vel=Q_vel,
        r_pos=20.0,          # sim: 0.08 m → 8.0 cm
        r_yaw=np.radians(15),       # sim: deg2rad(10) rad  (unchanged)
        prior=prior,
        init_std=init_std,
        radius=23.5,        # sim: 0.235 m → 23.5 cm
    )

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
        self.last_update_time = time.perf_counter()

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
        """Force re-initialisation on the next observation."""
        self.is_their_target_prev = False

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
        if not self.is_their_target_prev and panel.position is not None:
            prior = np.array(
                [panel.position[0], panel.position[1], 0, 0, panel.yaw, 0],
                dtype=np.float32,
            )
            self.pf.reinit(prior)
            # Reset clock so first dt isn't stale from before target acquisition
            self.last_update_time = self.ctx.frame_ts if self.ctx.frame_ts is not None else time.perf_counter()

        self.is_their_target_prev = True
        if self.ctx.new_observation:
            measurements = np.array(
                [[p.position[0], p.position[1], p.yaw]
                 for p in target_robot.panels if p.position is not None],
                dtype=np.float32,
            )  # shape (M, 3)
            dt = self.ctx.frame_ts - self.last_update_time
            self.last_update_time = self.ctx.frame_ts
            estimate, confidence = self.pf.update(dt, measurements)
        else:
            current_time = time.perf_counter()
            dt = current_time - self.last_update_time
            assert dt >= 0, f"Negative dt computed in estimation module: {dt:.4f}s (current_time={current_time:.4f}, last_update_time={self.last_update_time:.4f})"

            self.last_update_time = current_time
            estimate, confidence = self.pf.update_with_no_observation(dt)

        return RobotStateEstimate(
            value=estimate,
            timestamp=self.last_update_time,
            confidence=float(confidence),
            a_radius=23.5,
            b_radius=23.5,
        )

