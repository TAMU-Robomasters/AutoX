"""radii estimator for spinning robots using a Kalman filter."""

from typing import List, Optional

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

from src.core.module import Module, real
from src.types.autoaim import ArmorPanel, AutoAimContext


# Assumes only adjacent (90°-apart) panels are ever passed in.
class RadiiEstimatorModule(Module[AutoAimContext]):
    """estimates the (r_a, r_b) of the target robot's ellipse model.

    target is approximated as an ellipse with four panels every 90° from the spin centre c. 
    panels 1/3 lie on the major axis (r_a) and panels 2/4 on the minor axis (r_b).

    given two adjacent detected panels α and β with 2-D XZ positions p and
    inward-facing unit vectors û(θ) (section 2.1 of the state-estimation doc):

        c  = (p_β · û(θ_α)) û(θ_α) + (p_α · û(θ_β)) û(θ_β)
        r_α = |c − p_α|
        r_β = |c − p_β|

    measurements are smoothed by a kalman filter
    (F = I, H = I) whose noise params are taken from original code
    """

    def __init__(self, context: AutoAimContext):
        super().__init__(
            name="Radii Estimator",
            context=context,
            inputs=["panels"],
            outputs=["radii"],
        )
        self._kf = self._build_kf()
        self._initialized = False

    def _build_kf(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=2, dim_z=2)
        kf.F = np.eye(2)                          # constant-position model
        kf.H = np.eye(2)                          # measurement = state directly
        kf.P = np.diag([0.8 ** 2, 0.8** 2])     # initial covariance
        kf.Q = np.diag([0.001 ** 2, 0.001 ** 2]) # process noise
        kf.R = np.diag([0.5 ** 2, 0.5 ** 2])     # measurement noise
        kf.x = np.zeros((2, 1))
        return kf

    @real()
    def _run_estimate(self, panels: Optional[List[ArmorPanel]]) -> Optional[np.ndarray]:
        if not self._initialized and (panels is None or len(panels) < 2):
            return None

        self._kf.predict()

        if panels is not None and len(panels) >= 2:
            z = self._measure_radii(panels[0], panels[1])
            if z is not None:
                if not self._initialized:
                    self._kf.x = z.reshape(2, 1)
                    self._initialized = True
                self._kf.update(z)

        print("uncertainty:", self._kf.P[0, 0], self._kf.P[1, 1])
        return self._kf.x.flatten() if self._initialized else None

    def _measure_radii(
        self, panel_a: ArmorPanel, panel_b: ArmorPanel
    ) -> Optional[np.ndarray]:
        "compute r_α, r_β from two adjacent 90° panels, using 2.1 in the state-estimation document."
        if (
            panel_a.position is None or panel_a.orientation is None
            or panel_b.position is None or panel_b.orientation is None
        ):
            return None

        # 2-D panel positions in the XZ (horizontal) plane
        p_a = panel_a.position.flatten()[[0, 2]].astype(float) / 100
        p_b = panel_b.position.flatten()[[0, 2]].astype(float) / 100

        # Inward unit vectors û(θ_i) — point from each panel toward centre c
        R, _ = cv2.Rodrigues(panel_a.orientation)
        panel_yaw = (-np.arctan2(R[0, 2], R[2, 2])+np.deg2rad(180))
        print(f"panel_a yaw (degrees): {np.degrees(panel_yaw):.2f}")
        u_a = np.array([np.cos(panel_yaw), np.sin(panel_yaw)])  # from panel_a's orientation
        u_b = np.array([np.cos(panel_yaw + np.deg2rad(90)), np.sin(panel_yaw + np.deg2rad(90))])  # perpendicular to u_a

        # Robot centre estimate:  c = (p_β · û_α) û_α + (p_α · û_β) û_β
        c = (p_b @ u_a) * u_a + (p_a @ u_b) * u_b

        r_a = float(np.linalg.norm(c - p_a))
        r_b = float(np.linalg.norm(c - p_b))

        # always assign r_a to major axis and r_b to minor axis
        if r_a < r_b:
            r_a, r_b = r_b, r_a

        return np.array([r_a, r_b])

    def _inward_direction_xz(self, rvec: np.ndarray) -> np.ndarray:
        """return panel's inward unit vector in the 2-D XZ plane. """
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        inward_xz = -R[[0, 2], 2]   # negate outward normal, drop Y
        norm = np.linalg.norm(inward_xz)
        if norm < 1e-6:
            return np.array([1.0, 0.0])
        return inward_xz / norm
