"""Classical light-bar detector for the EKF engine: no per-frame PnP.

Same front end as ``ClassicalDetectorModule`` (threshold -> lights -> pairing ->
icon check) but it stops at pixels: each panel carries its two light bars'
corners (the EKF measurement), the icon, and ``range_cm``, a rough distance
from the light-bar pixel length used only to rank panels/robots by closeness
(classification sort, targeting, target switch). ``position``/``yaw`` stay
None; the EKF engine runs PnP itself, only when it needs to seed the filter
(``src/subsystems/estimation/ekf/pnp_seed.py``). Learner design, 2026-11-12.
"""

from typing import List, Optional

import numpy as np

from src.core.module import Module, real
from src.subsystems.display import CYAN, display
from src.subsystems.estimation.ekf.model import LIGHT_HEIGHT, PANEL_TILT
from src.subsystems.vision.classical_detector import (
    armor,
    frame_proccesing,
    icon_detection,
)
from src.toolbox.geometry_tools import BoundingBox
from src.toolbox.globals import config
from src.types.autoaim import ArmorPanel, FullStateAutoAimContext

METERS_TO_CM = 100.0


def light_bar_length_px(light_corners: np.ndarray) -> float:
    """Mean top-to-bottom length of a panel's two light bars, (2, 4, 2) TL,TR,BR,BL -> px."""
    top = (light_corners[:, 0] + light_corners[:, 1]) / 2.0
    bottom = (light_corners[:, 2] + light_corners[:, 3]) / 2.0
    return float(np.mean(np.linalg.norm(top - bottom, axis=1)))


def range_from_length_cm(length_px: float, fy: float) -> float:
    """Pinhole distance from light-bar length (face-on, panel tilt included); for ranking only."""
    return fy * LIGHT_HEIGHT * np.cos(PANEL_TILT) / length_px * METERS_TO_CM


class LightPanelDetectorModule(Module[FullStateAutoAimContext]):
    """Finds armor panels as light-bar pixel corners (no PnP)."""

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

        lights = armor.bounding_boxes(frame_proccesing.frame_process(frame), frame)
        if len(lights) <= 1:
            return None
        try:
            pairs = armor.pairing(lights)
        except Exception as e:  # noqa: BLE001 -- detector must not kill the loop
            self.log.warning("pairing failed: %s", e)
            return None

        panels: List[ArmorPanel] = []
        for pair in pairs:
            panel = armor.armour_corners(pair)
            if not icon_detection.icon_detection(panel, frame):
                continue
            corners = armor.light_corners(pair)
            length = light_bar_length_px(corners)
            if not np.isfinite(length) or length <= 1.0:
                continue
            outer = np.asarray(panel.corners, dtype=np.float64).reshape(-1, 2)
            display.windows["main"].add_contour(panel.inner_corners, color=CYAN)
            panels.append(
                ArmorPanel(
                    icon=getattr(panel, "id", None),
                    position=None,
                    orientation=None,
                    bbx=BoundingBox.from_points(
                        top_left=outer.min(axis=0), bottom_right=outer.max(axis=0)
                    ),
                    contour=None,
                    light_corners=corners,
                    range_cm=range_from_length_cm(length, self._fy),
                )
            )
        return panels
