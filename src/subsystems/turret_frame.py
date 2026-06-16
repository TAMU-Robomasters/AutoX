"""Shared helper for converting detected panels from the camera frame into
the turret / ballistic frame.

Both the particle-filter and the EKF auto-aim engines call this between the
detection and estimation stages, after they've asked the MCU for the
``camera_to_turret`` transform that corresponds to the frame's timestamp.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.types.autoaim import ArmorPanel

METERS_TO_CM = 100.0


def transform_panels_to_turret_frame(
    panels: Iterable[ArmorPanel],
    camera_to_turret_matrix: np.ndarray,
    turret_yaw: float,
) -> None:
    """Transform panel poses from camera frame to turret/ballistic frame in place.

    The classical detector already sets ``panel.yaw`` to the camera-relative
    panel yaw, so we just add the turret yaw to lift it into the global frame.
    Positions are converted cm → m for the 4x4 transform and back to cm so the
    rest of the pipeline can keep its cm units.
    """
    for panel in panels:
        if panel.position is None or panel.orientation is None:
            continue
        pos_m = np.append(panel.position / METERS_TO_CM, 1.0)
        panel.position = (camera_to_turret_matrix @ pos_m)[:3] * METERS_TO_CM
        panel.yaw = panel.yaw + turret_yaw
