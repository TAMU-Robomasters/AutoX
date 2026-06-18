"""Full-state shot-timing ballistic module.

Takes a ``RobotStateEstimate`` and computes the pitch, yaw, and alignment time
needed to hit the target. Intended for fast-spinning targets where the shooter
must wait for a panel to face the correct direction before firing.

Contract (turret frame; internal angles from +x CCW; output yaw in the MCU
convention -- see ballistics/solver.py):
    inputs:  ``estimate`` (must carry ``aim_z`` and per-parity radii),
             ``target_robot`` (presence gate only)
    outputs: ``solution`` (``is_confident=False`` = straight-line aim, hold fire)

Both the barrel **pitch/yaw** and the **alignment time** key off the panel on
the robot's *circumference* -- horizontal distance ``d - r_mean`` with
``r_mean = (a_radius + b_radius)/2``, on the turret->centre ray at ``aim_z`` --
solved via the shared ``solve_no_spin`` solver (the same one single_panel.py
uses). Aiming at the centre instead over-elevates for the extra ~r_mean of range
(the shot flies high), since the panel we actually time-and-hit is nearer. The
solver's travel time predicts the robot pose to the moment the bullet reaches
the panel surface, and that pose decides which panel faces the shooter and when
(``_alignment``).
"""
import time
from typing import Optional, Tuple

import numpy as np

from src.core.module import Module, mock, real
from src.subsystems.ballistics.solver import mcu_yaw_from_xy, solve_no_spin
from src.subsystems.estimation.filters import FullStateKF
from src.toolbox.globals import config
from src.types.autoaim import (
    BallisticSolution,
    EnemyRobot,
    FullStateAutoAimContext,
    RobotStateEstimate,
)

BALLISTIC = config.ballistic
METERS_TO_CM = 100.0

FEEDER_DELAY_S = -0.07 # firmware feeder actuation lead (s); tuned on-robot
_TRACK_ALIGNMENT_TIME_MS = 255


def _alignment(facing_angle: float, theta0: float, omega: float) -> Tuple[float, int]:
    """Time (s) until the next panel faces the shooter, and which panel it is.

    Panel ``k`` (angle ``theta0 + k*90deg``) is aligned when its outward normal
    reaches ``facing_angle``; the spin direction (sign of omega, CCW-positive)
    decides which panel arrives first.
    """
    thetas = theta0 + np.arange(4) * (np.pi / 2.0)
    if omega > 0:
        angle_to_travel = (facing_angle - thetas) % (2.0 * np.pi)
        k = int(np.argmin(angle_to_travel))
        return float(angle_to_travel[k] / omega), k
    if omega < 0:
        angle_to_travel = (thetas - facing_angle) % (2.0 * np.pi)
        k = int(np.argmin(angle_to_travel))
        return float(angle_to_travel[k] / -omega), k
    # Not spinning: the closest-to-aligned panel, no wait.
    misalignment = np.abs(
        np.arctan2(np.sin(thetas - facing_angle), np.cos(thetas - facing_angle))
    )
    return 0.0, int(np.argmin(misalignment))


class FullStateShotTimingModule(Module[FullStateAutoAimContext]):
    """Compute a shot-timed firing solution from the robot state estimate.

    For fast-spinning targets: computes when a panel will face the shooter and
    returns the alignment time so the embedded system gates the shot.
    """

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="full_state_shot_timing",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["solution"],
        )
        # Metre units for the shared LM solver (matches single_panel.py).
        self._g: float = -(abs(BALLISTIC.gravity) / METERS_TO_CM)      # m/s^2, negative
        self._v: float = BALLISTIC.projectile_velocity / METERS_TO_CM  # m/s
        self._barrel: np.ndarray = np.asarray(BALLISTIC.barrel_offset, dtype=float)  # m
        self._max_tof: float = float(BALLISTIC.max_time_of_flight)     # s
        self._max_range: float = float(BALLISTIC.max_range)            # cm
        self._tol: float = float(BALLISTIC.solver_tol)

    def _track_only(self, p_t_m: np.ndarray) -> BallisticSolution:
        """Straight-line aim with fire held (no real ballistic solution); p_t in metres."""
        return BallisticSolution(
            pitch=float(np.arctan2(p_t_m[2], np.hypot(p_t_m[0], p_t_m[1]))),
            yaw=mcu_yaw_from_xy(float(p_t_m[0]), float(p_t_m[1])),
            alignment_time_ms=_TRACK_ALIGNMENT_TIME_MS,
            is_confident=False,
        )

    @real()
    def _run_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Compute pitch, yaw, and alignment time from the estimate."""
        if estimate is None or target_robot is None:
            return None
        if estimate.aim_z is None:
            # Engine must not route here before parity anchoring sets aim_z.
            self.log.warning("estimate has no aim_z; cannot solve")
            return None

        est = estimate.value
        x, y = float(est[0]), float(est[1])
        vx, vy = float(est[2]), float(est[3])
        aim_z = float(estimate.aim_z)
        time_since_estimate = time.perf_counter() - estimate.timestamp

        p_center = np.array([x, y, aim_z]) / METERS_TO_CM  # metres
        v_t = np.array([vx, vy, 0.0]) / METERS_TO_CM

        d = np.hypot(x, y)  # cm
        if d > self._max_range:
            self.log.warning("target out of range (%.0fcm) -- tracking only", d)
            return self._track_only(p_center)

        # --- pitch/yaw + travel time: aim at the panel on the CIRCUMFERENCE ---
        # The shot lands on the panel timed to face us, which sits ~r_mean nearer
        # than the centre (same azimuth, on the turret->centre ray). Solving the
        # arc to the centre's range over-elevates for the extra distance and the
        # shot flies high, so aim pitch/yaw at the circumference point instead.
        r_mean = (estimate.a_radius + estimate.b_radius) / 2.0
        scale = max(d - r_mean, 0.0) / d if d > 1e-6 else 0.0
        p_circ = np.array([x * scale, y * scale, aim_z]) / METERS_TO_CM
        circ_sol = solve_no_spin(
            self._barrel, self._v, self._g, p_circ, v_t, tol=self._tol
        )
        if not circ_sol["success"] or not (0.0 < circ_sol["time"] < self._max_tof):
            self.log.warning(
                "no ballistic arc to circumference (residual=%.4f, t=%.2fs) -- tracking only",
                circ_sol["residual"],
                circ_sol["time"],
            )
            return self._track_only(p_circ)
        t_circ = circ_sol["time"]

        # Predict the robot pose at the moment the bullet reaches the panel surface
        # (constant-velocity extrapolation of the estimate -- no estimator instance).
        prediction = FullStateKF.predict_ahead(
            est, t_circ + time_since_estimate + FEEDER_DELAY_S
        )
        px, py = float(prediction[0]), float(prediction[1])
        theta0, omega = float(prediction[4]), float(prediction[5])
        facing_angle = float(np.arctan2(-py, -px))
        alignment_time, _panel_k = _alignment(facing_angle, theta0, omega)

        return BallisticSolution(
            pitch=float(circ_sol["pitch"]),
            yaw=float(circ_sol["yaw"]),  # MCU convention, from solve_no_spin
            alignment_time_ms=int(alignment_time * 1000),
        )

    @mock
    def _run_mock_solve(
        self, estimate: Optional[RobotStateEstimate], target_robot: Optional[EnemyRobot]
    ) -> Optional[BallisticSolution]:
        """Fixed plausible solution for mock mode."""
        return BallisticSolution(
            pitch=np.deg2rad(10),
            yaw=np.deg2rad(20),
            alignment_time_ms=500,
        )
