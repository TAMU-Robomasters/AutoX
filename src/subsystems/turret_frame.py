"""Shared helper for converting detected panels from the camera frame into
the turret / ballistic frame.

Called between detection and estimation, after asking the MCU for the
``camera_to_turret`` transform matching the frame's timestamp.

Both position AND the outward-normal yaw are carried through the SAME relabel +
camera->turret rotation, so they end up in one consistent frame. This mirrors
``full_state_autoaim._transform_panels_to_turret_frame`` -- NOT a scalar
``+ turret_yaw`` of the detector's camera-relative yaw, which is wrong for two
reasons: (1) ``panel.orientation`` (rvec) is still in RAW OpenCV camera axes
(x-right, y-down, z-fwd), unlike ``panel.position`` which pnp.py already
relabels to right/fwd/up via ``[v0, v2, -v1]``; and (2) the detector yaw is the
normal's azimuth about the optical axis in the camera's *pitched* plane, while
the ballistic frame wants it from +x in the level plane. So we relabel the 3-D
normal exactly like the position, push it through the same matrix rotation, and
read off its azimuth. The 4x4 ``camera_to_turret_matrix`` already encodes the
gimbal orientation, so no separate turret_yaw term is added.
"""

from __future__ import annotations

from typing import Iterable

import cv2 as cv
import numpy as np

from src.types.autoaim import ArmorPanel

METERS_TO_CM = 100.0


def transform_panels_to_turret_frame(
    panels: Iterable[ArmorPanel],
    camera_to_turret_matrix: np.ndarray,
) -> None:
    """Transform panel poses from camera frame to turret/ballistic frame in place.

    Positions are converted cm -> m for the 4x4 transform and back to cm so the
    rest of the pipeline keeps its cm units. ``panel.yaw`` is recomputed from the
    outward normal (rvec 3rd column) relabeled and rotated the same way.
    """
    R_ct = camera_to_turret_matrix[:3, :3]
    for panel in panels:
        if panel.position is None or panel.orientation is None:
            continue
        # tvec is in cm -> metres for the 4x4 transform, then back to cm.
        pos_m = np.append(panel.position / METERS_TO_CM, 1.0)
        panel.position = (camera_to_turret_matrix @ pos_m)[:3] * METERS_TO_CM

        # Panel +z (3rd column of R) is the outward normal in RAW OpenCV camera
        # axes. Relabel it to right/fwd/up just like pnp.py does to tvec, then
        # apply the same camera->turret rotation, and read off its azimuth.
        n_cam = cv.Rodrigues(panel.orientation)[0][:, 2]
        n_relabeled = np.array([n_cam[0], n_cam[2], -n_cam[1]])
        n_turret = R_ct @ n_relabeled
        panel.yaw = float(np.arctan2(n_turret[1], n_turret[0]))
