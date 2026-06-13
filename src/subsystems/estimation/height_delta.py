"""Inter-pair panel height-delta estimator (state-1 constant learning).

Contract (turret frame, cm -- see src/types/autoaim.py):
    inputs:  ``target_robot`` (panels transformed to the turret frame, ids
             assigned by PanelTrackingModule)
    outputs: ``height_delta_estimate`` (signed ``z_even - z_odd`` + variance;
             the engine's convergence gate reads the variance)

Same gate as RadiiEstimatorModule (exactly two adjacent-id panels ~90deg
apart); the measurement is simply the signed z difference between the
even-parity and odd-parity panels, smoothed by a scalar constant-state KF
(F = H = I) per robot name. The *sign* (which pair is lower) is the quantity
that anchors panel ids in state 2 -- relative, so platforms don't break it.

Statefulness: kept across target loss (static geometry; the engine handles
parity swaps); ``reset()`` clears it.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from src.core.module import Module, mock, real
from src.subsystems.estimation.robot_constants import measure_pair_by_parity
from src.types.autoaim import (
    EnemyRobot,
    FullStateAutoAimContext,
    HeightDeltaEstimate,
)

# Defaults (cm); same reasoning as radii.py.
_INIT_STD = 10.0
_PROCESS_STD = 0.05
_MEAS_STD = 2.0
_MAHALANOBIS_GATE = 16.0
_DZ_MAX = 30.0  # |z_even - z_odd| beyond this is a bad detection, not geometry


@dataclass
class _RobotHeightKF:
    kf: KalmanFilter
    n_updates: int = 0


def _build_kf() -> KalmanFilter:
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.eye(1)
    kf.H = np.eye(1)
    kf.P = np.array([[_INIT_STD**2]])
    kf.Q = np.array([[_PROCESS_STD**2]])
    kf.R = np.array([[_MEAS_STD**2]])
    kf.x = np.zeros((1, 1))
    return kf


class PanelHeightDeltaModule(Module[FullStateAutoAimContext]):
    """Estimate the signed height difference between the two panel pairs."""

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="height_delta_estimator",
            context=context,
            inputs=["target_robot"],
            outputs=["height_delta_estimate"],
        )
        self._filters: Dict[str, _RobotHeightKF] = {}

    def reset(self, robot_name: Optional[str] = None) -> None:
        """Discard learned state for one robot (or all when None)."""
        if robot_name is None:
            self._filters.clear()
        else:
            self._filters.pop(robot_name, None)

    @real()
    def _run_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[HeightDeltaEstimate]:
        if target_robot is None:
            return None

        entry = self._filters.get(target_robot.name)

        measurement = measure_pair_by_parity(target_robot.panels)
        if measurement is not None and abs(measurement.dz) < _DZ_MAX:
            z = measurement.dz  # z_even - z_odd, signed
            if entry is None:
                entry = _RobotHeightKF(kf=_build_kf())
                self._filters[target_robot.name] = entry
                entry.kf.x = np.array([[z]])  # seed from the first measurement
                entry.n_updates = 1
            else:
                entry.kf.predict()
                innovation = z - float(entry.kf.x[0, 0])
                S = float(entry.kf.P[0, 0]) + float(entry.kf.R[0, 0])
                if innovation**2 / S < _MAHALANOBIS_GATE:
                    entry.kf.update(np.array([z]))
                    entry.n_updates += 1
                else:
                    self.log.debug("height-delta outlier gated (dz=%.1f)", z)

        if entry is None:
            return None
        return HeightDeltaEstimate(
            dz=float(entry.kf.x[0, 0]),
            var=float(entry.kf.P[0, 0]),
            n_updates=entry.n_updates,
        )

    @mock
    def _run_mock_estimate(
        self, target_robot: Optional[EnemyRobot]
    ) -> Optional[HeightDeltaEstimate]:
        """Fixed plausible height delta for mock mode."""
        return HeightDeltaEstimate(dz=0.0, var=0.01, n_updates=100)
