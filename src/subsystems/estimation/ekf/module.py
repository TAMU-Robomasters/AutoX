"""Pipeline module wrapping the full-robot EKF for the EKF auto-aim engine.

Contract:
    inputs:  ``target_robot`` (its panels must carry ``light_corners``, and
             ``position``/``yaw`` already transformed to the turret frame)
    reads (engine-written): ``new_observation``, ``frame_ts``, ``camera_to_turret``
    outputs: ``ekf_estimate`` (raw 10-state, metres) and ``estimate`` (the same
             state remapped to AutoX's ``RobotStateEstimate`` so the existing
             ``FullStateShotTimingModule`` can consume it)

On a fresh observation: seed if uninitialised (two adjacent panels when both
are seen, else the closest one; no update on the seed frame), else predict to
the frame timestamp and update with every target panel. If no update has been
accepted for ``LOST_TIME_S`` the filter is reset and re-seeded from the current
detection (awakening's TRACKING -> LOST -> re-init rule). On other ticks it
re-publishes the last state untouched; the ballistic modules extrapolate from
``timestamp`` themselves. (No detections at all for 500 ms is handled by the
engine's ``target_reset_timeout_ms``, which calls ``reset()``.)
"""

from typing import List, Optional, Tuple

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.estimation.ekf import model
from src.subsystems.estimation.ekf.filter import FullRobotEKF
from src.subsystems.estimation.ekf.model import (
    H, OMEGA, THETA, VX, VY, X, Y, ZA,
)
from src.types.autoaim import ArmorPanel, EkfStateEstimate, EnemyRobot, RobotStateEstimate

METERS_TO_CM = 100.0
# Reset + re-seed when no update has been accepted for this long (learner,
# 2026-11-11; awakening's lost_time_thres, 0.5 s as in Full-Robot-EKF).
LOST_TIME_S = 0.5


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
        self._last_ts: Optional[float] = None
        self._last_accept_ts: Optional[float] = None
        self.n_resets = 0
        self.seed_mode: Optional[str] = None

    def initialize(self) -> None:
        """Load the AutoX camera intrinsics (child process; mrcal on the robot)."""
        if self._K is None or self._dist is None:
            from src.drivers.video_stream import camera_info

            intr = camera_info.intrinsics()
            self._K = intr.camera_matrix
            self._dist = intr.distortion_coefficients

    def reset(self) -> None:
        self.ekf.reset()
        self._last_ts = None
        self._last_accept_ts = None

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
        usable = [
            p for p in panels
            if p.light_corners is not None and p.position is not None
        ]
        if not usable:
            return
        cam = model.Camera(self._K, self._dist, cam_to_turret)

        if (self.ekf.initialized and self._last_accept_ts is not None
                and frame_ts - self._last_accept_ts > LOST_TIME_S):
            self.log.warning("EKF: no accepted update for %.2f s; reset + re-seed",
                             frame_ts - self._last_accept_ts)
            self.ekf.reset()
            self.n_resets += 1

        if not self.ekf.initialized:
            self._seed(usable)
            self._last_ts = self._last_accept_ts = frame_ts
            return                      # no update on the seed frame

        dt = frame_ts - self._last_ts if self._last_ts is not None else 0.0
        self.ekf.predict(dt)
        self._last_ts = frame_ts
        if self.ekf.update([p.light_corners for p in usable], cam).accepted:
            self._last_accept_ts = frame_ts

    def _seed(self, panels: List[ArmorPanel]) -> None:
        """Two adjacent panels -> ``seed_pair`` (closest = panel 0); else closest panel."""
        ordered = sorted(panels, key=lambda p: float(np.linalg.norm(p.position)))
        pos = [np.asarray(p.position, dtype=np.float64) / METERS_TO_CM for p in ordered]
        for i in range(1, len(ordered)):
            if self.ekf.seed_pair(pos[0], ordered[0].yaw, pos[i], ordered[i].yaw):
                self.seed_mode = "two_panels"
                return
        self.ekf.seed(pos[0], ordered[0].yaw)
        self.seed_mode = "one_panel"

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
