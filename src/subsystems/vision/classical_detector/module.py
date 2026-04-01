"""Classic vision processing module for auto-aim pipeline.

Capable of finding position, orientation, and icon of armor panels.
"""

from typing import List, Optional

import numpy as np

from src.core.module import Context, Module, real
from src.subsystems.display import CYAN, display
from src.subsystems.vision.classical_detector import (
    armor,
    frame_proccesing,
    icon_detection,
    pnp,
)
from src.toolbox.geometry_tools import BoundingBox
from src.types.autoaim import ArmorPanel


def _process_pairs(pairs, frame) -> List[ArmorPanel]:
    """Filter light pairs into validated ArmorPanel instances."""
    panels: List[ArmorPanel] = []
    for pair in pairs:
        panel = armor.armour_corners(pair)
        if not icon_detection.icon_detection(panel, frame):
            continue
        try:
            pnp.get_cord(panel)
        except Exception as e:
            print(f"Error in get_cord for panel: {e}")
            continue
        # remove panels that are yawed too much
        if abs(np.degrees(panel.rvec[2])) < 40:
            panels.append(
                ArmorPanel(
                    icon=panel.id if hasattr(panel, "id") else None,
                    position=panel.tvec if hasattr(panel, "tvec") else None,
                    orientation=panel.rvec if hasattr(panel, "rvec") else None,
                    bbx=BoundingBox.from_points(
                        top_left=panel.corners[0][0],
                        bottom_right=panel.corners[2][0],
                    ),
                    contour=panel.corners,
                )
            )
    return panels


class ClassicalDetectorModule(Module[Context]):
    """Detects armor panels using classical computer-vision techniques."""

    def __init__(self, context: Context):
        super().__init__(
            name="classical_detector",
            context=context,
            inputs=[],
            outputs=["panels"],
        )

    @real()
    def _run_detect(self) -> Optional[List[ArmorPanel]]:
        """Process the current video frame and populate *ctx.panels*."""
        frame = self.ctx.frame
        assert frame is not None, "No frame received."


        contours = frame_proccesing.frame_process(frame)
        lights = armor.bounding_boxes(contours, frame)

        # Not enough lights to form a panel
        if len(lights) <= 1:
            panels = None
            return panels

        try:
            pairs = armor.pairing(lights)
        except Exception as e:
            print(f"Error in pairing: {e}")
            panels = None
            return panels

        panels = _process_pairs(pairs, frame)
        for panel in panels:
            display.windows["main"].add_contour(panel.contour, color=CYAN)
        return panels
