"""Per-parity panel-orbit radii estimator (state-1 constant learning).

Contract (turret frame, cm, angles from +x CCW -- see src/types/autoaim.py):
    inputs:  ``target_robot`` (panels transformed to the turret frame, ids
             assigned by PanelTrackingModule)
    outputs: ``radii_estimate`` (per-parity radii + variances; the engine's
             convergence gate reads the variances)

Whenever exactly two adjacent-id panels are visible, both orbit radii are
measured at once via the orthogonal-projection centre construction
(``measure_adjacent_pair``: ``c = (p_b . u_a) u_a + (p_a . u_b) u_b``) and
smoothed by a constant-state Kalman filter (F = I, H = I) over
``[r_even, r_odd]`` -- keyed by panel-id *parity*, never sorted by size:
the pairing with panel ids is the whole point. A Mahalanobis gate drops
outlier measurements (id misassignments near the 45deg tracking boundary).

Statefulness: one KF per robot name, kept across target loss (the radii are
static geometry; the engine handles the parity-swap risk via
``match_parity`` + the tracker's parity offset). ``reset()`` clears it
(``force_reestimation``).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.stats import chi2

from src.core.module import Module, mock, real
from src.subsystems.estimation.robot_constants import measure_pair_by_parity
from src.types.autoaim import (
    EnemyRobot,
    FullStateAutoAimContext,
    RadiiEstimate,
)

# Defaults (cm). The robot's radii are fixed geometry, so process noise is
# tiny; measurement noise reflects PnP + transform error on a single frame.
_INIT_STD = 10.0      # prior std on each radius
_PROCESS_STD = 0.05   # per-update drift allowance
_MEAS_STD = 3.0       # single-measurement noise
_GATE_DOF = 2         # innovation is [r_even, r_odd]; chi-squared dof for the gate
_DEFAULT_GATE_CONFIDENCE = 0.9997  # ~= a fixed gate of 16.0 for 2 dof
_R_MIN, _R_MAX = 5.0, 45.0  # physically plausible orbit radii (cm)


@dataclass
class _RobotRadiiKF:
    kf: KalmanFilter
    n_updates: int = 0


def _build_kf(initial_guess: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.F = np.eye(2)                       # constant-state model
    kf.H = np.eye(2)                       # measurement = state directly
    kf.P = np.diag([_INIT_STD**2] * 2)
    kf.Q = np.diag([_PROCESS_STD**2] * 2)
    kf.R = np.diag([_MEAS_STD**2] * 2)
    kf.x = np.full((2, 1), initial_guess)
    return kf


class RadiiEstimatorModule(Module[FullStateAutoAimContext]):
    """Estimate the (r_even, r_odd) panel-orbit radii of the target robot."""

    def __init__(
        self,
        context: FullStateAutoAimContext,
        initial_guess: float = 23.5,
        gate_confidence: float = _DEFAULT_GATE_CONFIDENCE,
    ):
        super().__init__(
            name="radii_estimator",
            context=context,
            inputs=["target_robot"],
            outputs=["radii_estimate"],
        )
        self._initial_guess = float(initial_guess)
        # Mahalanobis (chi-squared, 2 dof) outlier gate, set from a confidence
        # percentile so it can be tuned in physical terms from config.
        self._mahalanobis_gate = float(chi2.ppf(gate_confidence, _GATE_DOF))
        self._filters: Dict[str, _RobotRadiiKF] = {}

    def reset(self, robot_name: Optional[str] = None) -> None:
        """Discard learned state for one robot (or all when None)."""
        if robot_name is None:
            self._filters.clear()
        else:
            self._filters.pop(robot_name, None)

    @real()
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RadiiEstimate]:
        if target_robot is None:
            return None

        entry = self._filters.get(target_robot.name)

        measurement = measure_pair_by_parity(target_robot.panels)
        if measurement is not None:
            z = np.array([measurement.r_a, measurement.r_b])  # [r_even, r_odd]
            if np.all((z > _R_MIN) & (z < _R_MAX)):
                if entry is None:
                    entry = _RobotRadiiKF(kf=_build_kf(self._initial_guess))
                    self._filters[target_robot.name] = entry
                    entry.kf.x = z.reshape(2, 1)  # seed from the first measurement
                    entry.n_updates = 1
                else:
                    entry.kf.predict()
                    innovation = z - entry.kf.x.flatten()
                    S = entry.kf.P + entry.kf.R
                    mahalanobis_sq = float(innovation @ np.linalg.solve(S, innovation))
                    if mahalanobis_sq < self._mahalanobis_gate:
                        entry.kf.update(z)
                        entry.n_updates += 1
                    else:
                        self.log.debug(
                            "radii outlier gated (m2=%.1f, z=%s)", mahalanobis_sq, z
                        )

        if entry is None:
            return None
        return RadiiEstimate(
            r_even=float(entry.kf.x[0, 0]),
            r_odd=float(entry.kf.x[1, 0]),
            var_even=float(entry.kf.P[0, 0]),
            var_odd=float(entry.kf.P[1, 1]),
            n_updates=entry.n_updates,
        )

    @mock
    def _run_mock_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[RadiiEstimate]:
        """Fixed plausible radii for mock mode."""
        return RadiiEstimate(
            r_even=23.5, r_odd=23.5, var_even=0.01, var_odd=0.01, n_updates=100
        )
