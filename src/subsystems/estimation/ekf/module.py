"""Pipeline module wrapping the full-robot EKF for the EKF auto-aim engine.

Contract:
    inputs:  ``target_robot`` (its panels must carry ``light_corners``; pixels)
    reads (engine-written): ``new_observation``, ``frame_ts``, ``camera_to_turret``
    outputs: ``ekf_estimate`` (raw 10-state, metres) and ``estimate`` (the same
             state remapped to AutoX's ``RobotStateEstimate`` so the existing
             ``FullStateShotTimingModule`` can consume it)

Filter steps only (learner, 2026-11-12: wide logic lives in the engine). On a
fresh observation, if the filter is seeded and not already at this frame,
predict to the frame timestamp and update with every target panel; otherwise
re-publish the last state untouched (the ballistic modules extrapolate from
``timestamp`` themselves). WHEN to seed or reset -- PnP seed, 0.5 s lost reset,
target loss/switch -- is decided by ``EkfAutoAimEngine``, which calls
``seed()`` / ``reset()``. A seed sets the filter to the seed frame, so that
frame is not also used as an update.
"""

from typing import List, Optional, Tuple

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.estimation.ekf import model
from src.subsystems.estimation.ekf.filter import FullRobotEKF
from src.subsystems.estimation.ekf.model import (
    OMEGA,
    THETA,
    VX,
    VY,
    ZA,
    H,
    X,
    Y,
)
from src.subsystems.estimation.ekf.pnp_seed import Seed
from src.types.autoaim import (
    ArmorPanel,
    EkfStateEstimate,
    EnemyRobot,
    RobotStateEstimate,
)

METERS_TO_CM = 100.0


def to_robot_state_estimate(est: EkfStateEstimate) -> RobotStateEstimate:
    """EKF state (m) -> AutoX ``RobotStateEstimate`` (cm) for the shot-timing module.

    ``value = [x, y, vx, vy, theta, omega, aim_z]``; aim_z is the mid-height of
    the two panel pairs (z_a + h/2). Panel convention is identical (k at
    theta + k*90deg, even = a_radius, odd = b_radius), so no re-indexing.
    """
    s = est.x
    ra, rb = model.radii(s)
    aim_z = (s[ZA] + 0.5 * s[H]) * METERS_TO_CM
    value = np.array([
        s[X] * METERS_TO_CM, s[Y] * METERS_TO_CM,
        s[VX] * METERS_TO_CM, s[VY] * METERS_TO_CM,
        s[THETA], s[OMEGA], aim_z,
    ])
    return RobotStateEstimate(
        value=value,
        timestamp=est.timestamp,
        confidence=1.0,
        a_radius=ra * METERS_TO_CM,
        b_radius=rb * METERS_TO_CM,
        aim_z=aim_z,
    )


class EkfEstimationModule(Module):
    """Runs the full-robot EKF on the target robot's panels."""

    def __init__(
        self,
        context,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            name="ekf_estimation",
            context=context,
            inputs=["target_robot"],
            outputs=["ekf_estimate", "estimate"],
        )
        self.ekf = FullRobotEKF()
        self._K = camera_matrix
        self._dist = dist_coeffs
        self._last_ts: Optional[float] = None       # frame the filter is at
        self.last_accept_ts: Optional[float] = None  # last accepted update (or seed)

    def initialize(self) -> None:
        """Load the AutoX camera intrinsics (child process; mrcal on the robot)."""
        if self._K is None or self._dist is None:
            from src.drivers.video_stream import camera_info

            intr = camera_info.intrinsics()
            self._K = intr.camera_matrix
            self._dist = intr.distortion_coefficients

    def reset(self) -> None:
        """Back to uninitialised (engine-decided)."""
        self.ekf.reset()
        self._last_ts = None
        self.last_accept_ts = None

    def seed(self, seed: Seed, frame_ts: float) -> None:
        """Initialise the filter at ``frame_ts`` (engine-decided)."""
        seed.apply(self.ekf)
        self._last_ts = self.last_accept_ts = frame_ts

    @property
    def intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """(camera_matrix, dist_coeffs) loaded in ``initialize``."""
        return self._K, self._dist

    # ------------------------------------------------------------------

    @real()
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Tuple[Optional[EkfStateEstimate], Optional[RobotStateEstimate]]:
        ctx = self.ctx
        if target_robot is not None and ctx.new_observation and ctx.camera_to_turret is not None:
            self._observe(target_robot.panels, ctx.frame_ts, ctx.camera_to_turret)
        return self._outputs_now()

    def _observe(self, panels: List[ArmorPanel], frame_ts: float, cam_to_turret: np.ndarray) -> None:
        if not self.ekf.initialized or self._last_ts is None or frame_ts <= self._last_ts:
            return
        corners = [p.light_corners for p in panels if p.light_corners is not None]
        if not corners:
            return
        self.ekf.predict(frame_ts - self._last_ts)
        self._last_ts = frame_ts
        cam = model.Camera(self._K, self._dist, cam_to_turret)
        if self.ekf.update(corners, cam).accepted:
            self.last_accept_ts = frame_ts

    def _outputs_now(self) -> Tuple[Optional[EkfStateEstimate], Optional[RobotStateEstimate]]:
        if not self.ekf.initialized or self._last_ts is None:
            return None, None
        est = EkfStateEstimate(
            x=self.ekf.x.copy(),
            timestamp=self._last_ts,
            accepted_updates=self.ekf.accepted_count,
        )
        return est, to_robot_state_estimate(est)

    @mock
    def _run_mock_estimate(self, target_robot):
        return None, None
