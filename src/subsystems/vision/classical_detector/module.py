"""Classic vision processing module for auto-aim pipeline.

Capable of finding position, orientation, and icon of armor panels.
"""
from src.subsystems.video_streaming.video_stream import video_stream

from typing import List, Optional

import cv2 as cv
import numpy as np

from src.core.module import Module, real
from src.subsystems.display import CYAN, GREEN, display, BLUE
from src.subsystems.vision.classical_detector import (
    armor,
    frame_proccesing,
    icon_detection,
    pnp,
)
from src.toolbox.geometry_tools import BoundingBox
from src.toolbox.globals import config
from src.types.autoaim import ArmorPanel, FullStateAutoAimContext


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
        # solvePnP may have failed silently; outer_corners is only set on success.
        if panel.outer_corners is None or panel.rvec is None:
            continue
        # Camera-relative panel yaw: 0 = panel face square to camera,
        # ±90° = edge-on. rvec is Rodrigues axis*angle, so its z-component is
        # not itself the yaw angle — extract via the rotation matrix instead.
        # The PF engine adds turret_yaw on top of this to get global yaw.
        R, _ = cv.Rodrigues(panel.rvec)
        yaw_rad = -np.arctan2(R[0, 2], R[2, 2]) + np.pi
        yaw_rad = (yaw_rad + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
        if abs(np.degrees(yaw_rad)) < 90:
            # Inner light-bar box (what we actually fed PnP) in cyan;
            # reprojected outer panel boundary in green.
            display.windows["main"].add_contour(panel.inner_corners, color=CYAN)
            display.windows["main"].add_contour(panel.outer_corners, color=BLUE)
            panels.append(
                ArmorPanel(
                    icon=panel.id if hasattr(panel, "id") else None,
                    position=panel.tvec if hasattr(panel, "tvec") else None,
                    orientation=panel.rvec if hasattr(panel, "rvec") else None,
                    bbx=BoundingBox.from_points(
                        top_left=panel.outer_corners[0][0],
                        bottom_right=panel.outer_corners[2][0],
                    ),
                    contour=panel.outer_corners,
                    yaw=float(yaw_rad),
                )
            )
    return panels


class ClassicalDetectorModule(Module[FullStateAutoAimContext]):
    """Detects armor panels using classical computer-vision techniques."""

    def __init__(self, context: FullStateAutoAimContext):
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
        if frame is None:
            # Transitional fallback for engines not yet cut over to feeding
            # ctx.frame from a CameraSource/FrameReader (e.g. the PF engine).
            f = video_stream.get_frame()
            frame = np.asarray(f.data)  # f is a Frame at runtime; no-op, no copy
            self.ctx.frame_ts = f.timestamp
        assert frame is not None, "No frame received."
        if config.log.display_live_frames:
            # set_image copies (so read-only shm frames are safe to draw on);
            # skip it entirely when we're not showing anything.
            display.windows["main"].set_image(frame)


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

        # _process_pairs handles inner+outer contour drawing inline so we keep
        # the Panel object's inner_corners in scope. No further display calls needed.
        panels = _process_pairs(pairs, frame)
        return panels
