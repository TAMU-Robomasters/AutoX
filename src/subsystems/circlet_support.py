"""Geometry + classification helpers for circlet detections (no detector/mrcal).

Kept separate from ``src/engines/circlet.py`` (which imports the classical
detector, and through it mrcal) so the pure transform/classification logic is
importable and unit-testable with no camera, no mrcal, no iceoryx2.

Frames (``src/types/autoaim.py`` convention, cm, x right / y forward / z up):
- A circlet camera produces panels in its **camera optical frame** (already
  relabeled to right/fwd/up by ``pnp.py``).
- :func:`panel_to_chassis` applies the camera's static **camera->chassis**
  extrinsic -> a :class:`CircletPanel` in the chassis frame (what gets published).
- :func:`circlet_panels_to_robots` lifts chassis -> turret with the live gimbal
  pose and buckets by icon, so the auto-aim engine can fold them into target
  selection.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2 as cv
import numpy as np

from src.types.autoaim import (
    ICON_TO_ROBOT_NAME,
    ArmorPanel,
    EnemyRobot,
)
from src.types.circlet import CircletDetections, CircletPanel

METERS_TO_CM = 100.0


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def extrinsic_from_cfg(
    yaw_deg: float = 0.0, translation_cm=(0.0, 0.0, 0.0)
) -> np.ndarray:
    """Build a camera->chassis 4x4 from a mounting yaw + translation.

    A body-fixed ring camera is modeled as a yaw rotation about the chassis z
    plus a fixed offset. Defaults give the identity (single-camera bring-up
    before real per-camera calibration). Translation is in cm; the matrix stores
    it in metres to match :func:`panel_to_chassis`'s cm<->m scaling.
    """
    e = np.eye(4)
    e[:3, :3] = _rot_z(np.radians(float(yaw_deg)))
    e[:3, 3] = np.asarray(translation_cm, dtype=float) / METERS_TO_CM
    return e


def panel_to_chassis(
    panel: ArmorPanel, extrinsic: np.ndarray, camera_id: int, timestamp: float
) -> CircletPanel:
    """Transform a detected panel into the chassis frame (a publishable CircletPanel).

    Mirrors ``_transform_panels_to_turret_frame`` in the auto-aim engine: the
    position (cm) goes through the 4x4 in metres; the outward normal is relabeled
    from the raw OpenCV rvec and rotated, then read off as a chassis-frame azimuth.
    """
    rot = extrinsic[:3, :3]
    pos_m = np.append(np.asarray(panel.position, dtype=float) / METERS_TO_CM, 1.0)
    pos_chassis = (extrinsic @ pos_m)[:3] * METERS_TO_CM

    yaw = float(panel.yaw)
    if panel.orientation is not None:
        n_cam = cv.Rodrigues(np.asarray(panel.orientation, dtype=float))[0][:, 2]
        n_relabeled = np.array([n_cam[0], n_cam[2], -n_cam[1]])
        n_chassis = rot @ n_relabeled
        yaw = float(np.arctan2(n_chassis[1], n_chassis[0]))

    return CircletPanel(
        icon=panel.icon,
        position=pos_chassis,
        yaw=yaw,
        camera_id=camera_id,
        timestamp=timestamp,
    )


def chassis_to_turret_matrix(turret_yaw: float, turret_pitch: float) -> np.ndarray:
    """Chassis->turret rotation from the gimbal yaw/pitch (3x3).

    STUB / first approximation: the turret is yawed then pitched relative to the
    chassis, so a chassis-frame vector maps to the turret frame by the inverse
    rotation ``Rx(-pitch) @ Rz(-yaw)`` (origins assumed coincident -- good enough
    for coarse 360 target *selection*). The exact axis order / signs depend on the
    gimbal convention and MUST be validated on hardware before this drives firing.
    """
    return _rot_x(-float(turret_pitch)) @ _rot_z(-float(turret_yaw))


def circlet_panels_to_robots(
    detections: CircletDetections, turret_rotation: Optional[np.ndarray] = None
) -> Dict[str, EnemyRobot]:
    """Lift circlet panels chassis->turret and bucket them into robots by icon.

    Returns ``{robot_name: EnemyRobot}`` only for robots that got at least one
    panel, each robot's panels sorted nearest-first (matching
    ``RobotClassificationModule``). ``turret_rotation`` is the 3x3 from
    :func:`chassis_to_turret_matrix`; ``None`` leaves panels in the chassis frame
    (identity -- useful for tests / a chassis-only consumer).
    """
    robots: Dict[str, EnemyRobot] = {}
    for cp in detections.panels:
        if cp.icon is None:
            continue
        name = ICON_TO_ROBOT_NAME.get(int(cp.icon))
        if name is None:
            continue
        pos = np.asarray(cp.position, dtype=float)
        if turret_rotation is not None:
            pos = turret_rotation @ pos
        panel = ArmorPanel(
            icon=cp.icon,
            position=pos,
            orientation=None,
            bbx=None,
            contour=None,
            yaw=cp.yaw,
        )
        robots.setdefault(name, EnemyRobot(name=name)).panels.append(panel)

    for robot in robots.values():
        # position is always set here (built just above); guard narrows for mypy.
        robot.panels.sort(
            key=lambda p: float(
                np.linalg.norm(p.position if p.position is not None else 0.0)
            )
        )
    return robots
