"""Classical light-bar detector for the EKF engine: no PnP, no icon check.

Light extraction follows Full-Robot-EKF ``frame_process`` (no-decay branch):

  1. colour mask: red -> R channel >= red_thresh[0]; blue -> (B - R) >= 120
  2. 3x3 morphological close, contours
  3. shape filter: area > 20 px, bounding box taller than wide, minAreaRect
     short/long ratio in (0.001, 0.30), long side within 40 deg of vertical
  4. corners ordered TL/TR/BR/BL along the bar's long axis

Lights are then paired into panels with AutoX's scored ``armor.pairing``
(the EKF update is panel-level). Each panel carries its two light bars'
corners (the EKF measurement) and ``range_cm``, a rough distance from the
light-bar pixel length used only to rank panels by closeness. There is NO icon
check (learner, 2026-11-13: it rejected 94% of panels on the aerial rig), so
panels have no robot identity -- the EKF engine treats them as one target.
``position``/``yaw`` stay None; the engine runs PnP only to seed the filter.

Colour: ``config.our_team_color`` names the colour that is DETECTED (same
convention as the old ``frame_proccesing.frame_process``).
"""

from typing import List, Optional

import cv2 as cv
import numpy as np

from src.core.module import Module, real
from src.subsystems.display import display
from src.subsystems.estimation.ekf.model import LIGHT_HEIGHT, PANEL_TILT
from src.subsystems.vision.classical_detector import armor
from src.toolbox.geometry_tools import BoundingBox
from src.toolbox.globals import config
from src.types.autoaim import ArmorPanel, FullStateAutoAimContext

METERS_TO_CM = 100.0

# Shape filter (Full-Robot-EKF / awakening defaults).
MIN_AREA_PX = 20.0
LIGHT_MIN_WH_RATIO = 0.001     # short / long: rejects dots
LIGHT_MAX_WH_RATIO = 0.30      # short / long: rejects fat blobs
LIGHT_MAX_TILT_DEG = 40.0      # long side from vertical
MASK_UPPER = 255
BLUE_DIFF_LOWER = 120          # (B - R) lower bound, Full-Robot-EKF default; red uses
                               # config.classical.red_thresh[0] (the old detector's threshold)


def colour_mask(frame: np.ndarray, colour: str, lower: int) -> np.ndarray:
    """Binary mask of the detected colour's light bars."""
    if colour == "red":
        return cv.inRange(frame[:, :, 2], lower, MASK_UPPER)
    if colour == "blue":
        return cv.inRange(cv.subtract(frame[:, :, 0], frame[:, :, 2]), lower, MASK_UPPER)
    raise ValueError(f"our_team_color must be 'red' or 'blue', got {colour!r}")


def is_light(rect) -> bool:
    """Plausible light bar: short/long ratio and tilt from vertical (from the box corners)."""
    (_, _), (w, h), _ = rect
    if w <= 0 or h <= 0:
        return False
    ratio = min(w, h) / max(w, h)
    if not LIGHT_MIN_WH_RATIO < ratio < LIGHT_MAX_WH_RATIO:
        return False
    box = cv.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    return float(np.degrees(np.arctan2(abs(e[0]), abs(e[1])))) <= LIGHT_MAX_TILT_DEG


def _to_light(rect) -> armor.Lights:
    """minAreaRect -> armor.Lights in the angle/size convention ``armor.pairing`` expects."""
    (cx, cy), (rw, rh), angle = rect
    w, h = (rh, rw) if rw > rh else (rw, rh)          # h = long side
    if rw > rh:
        angle += 90
    return armor.Lights(cx, cy, w, h, angle, rect=rect)


def find_lights(frame: np.ndarray, colour: str, lower: int) -> List[armor.Lights]:
    """Full-Robot-EKF light extraction."""
    closing = cv.morphologyEx(colour_mask(frame, colour, lower), cv.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    contours, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    lights = []
    for c in contours:
        if cv.contourArea(c) <= MIN_AREA_PX:
            continue
        _, _, bw, bh = cv.boundingRect(c)
        if bw > bh:
            continue
        rect = cv.minAreaRect(c)
        if is_light(rect):
            lights.append(_to_light(rect))
    return lights


def light_bar_length_px(light_corners: np.ndarray) -> float:
    """Mean top-to-bottom length of a panel's two light bars, (2, 4, 2) TL,TR,BR,BL -> px."""
    top = (light_corners[:, 0] + light_corners[:, 1]) / 2.0
    bottom = (light_corners[:, 2] + light_corners[:, 3]) / 2.0
    return float(np.mean(np.linalg.norm(top - bottom, axis=1)))


def range_from_length_cm(length_px: float, fy: float) -> float:
    """Pinhole distance from light-bar length (face-on, panel tilt included); for ranking only."""
    return fy * LIGHT_HEIGHT * np.cos(PANEL_TILT) / length_px * METERS_TO_CM


def detect_panels(frame: np.ndarray, colour: str, lower: int, fy: float) -> List[ArmorPanel]:
    """Lights -> scored pairs -> panels (closest first)."""
    lights = find_lights(frame, colour, lower)
    if len(lights) < 2:
        return []
    panels = []
    for pair in armor.pairing(lights):
        corners = armor.light_corners(pair)
        length = light_bar_length_px(corners)
        if not np.isfinite(length) or length <= 1.0:
            continue
        pts = corners.reshape(-1, 2)
        panels.append(ArmorPanel(
            icon=None, position=None, orientation=None,
            bbx=BoundingBox.from_points(top_left=pts.min(axis=0), bottom_right=pts.max(axis=0)),
            contour=None, light_corners=corners, range_cm=range_from_length_cm(length, fy),
        ))
    panels.sort(key=lambda p: p.distance_cm())
    return panels


class LightPanelDetectorModule(Module[FullStateAutoAimContext]):
    """Finds armor panels as light-bar pixel corners (no PnP, no icon check)."""

    def __init__(self, context: FullStateAutoAimContext, fy: Optional[float] = None):
        super().__init__(
            name="light_panel_detector",
            context=context,
            inputs=[],
            outputs=["panels"],
        )
        self._fy = fy

    def initialize(self) -> None:
        """Load the camera focal length (child process; mrcal on the robot)."""
        if self._fy is None:
            from src.drivers.video_stream import camera_info

            self._fy = float(camera_info.intrinsics().camera_matrix[1, 1])

    @real()
    def _run_detect(self) -> Optional[List[ArmorPanel]]:
        frame = self.ctx.frame
        assert frame is not None, "No frame received."
        if config.log.display_live_frames:
            display.windows["main"].set_image(frame)
        colour = config.our_team_color
        lower = int(config.classical.red_thresh[0]) if colour == "red" else BLUE_DIFF_LOWER
        return detect_panels(frame, colour, lower, self._fy) or None
