"""Depth-camera variant of the classical detector.

Identical panel detection and PnP to :class:`ClassicalDetectorModule`, with one
difference: the panel *position* (tvec) is taken from the depth camera —
``video_stream.get_xyz_at`` at the panel-center pixel — instead of from PnP.
Orientation (``rvec`` / yaw) still comes from PnP. Intended for RealSense-style
cameras (``config.hardware.camera_has_depth``).
"""

from typing import List, Optional

import cv2 as cv
import numpy as np

from src.core.module import Context, Module, real
from src.subsystems.display import BLUE, CYAN, display
from src.subsystems.vision.classical_detector import (
    armor,
    frame_proccesing,
    icon_detection,
    pnp,
)
from src.toolbox.geometry_tools import BoundingBox
from src.types.autoaim import ArmorPanel

# get_xyz_at returns metres; panel.position is consumed in cm
# (see particle_filter_autoaim.py: pos_m = panel.position / METERS_TO_CM).
METERS_TO_CM = 100


def _process_pairs(pairs, frame) -> List[ArmorPanel]:
    """Filter light pairs into ArmorPanels, taking position from depth."""
    # Lazy import: the legacy singleton opens a camera stream at import time, so
    # importing it at module load would break CAMERA=NONE (mock/sim) paths.
    from src.subsystems.video_streaming.video_stream import video_stream

    panels: List[ArmorPanel] = []
    for pair in pairs:
        panel = armor.armour_corners(pair)
        if not icon_detection.icon_detection(panel, frame):
            continue
        try:
            # PnP gives us rvec (orientation) and the reprojected outer corners;
            # we deliberately ignore its tvec and use depth for position.
            pnp.get_cord(panel)
        except Exception as e:
            print(f"Error in get_cord for panel: {e}")
            continue
        # solvePnP may have failed silently; outer_corners is only set on success.
        if panel.outer_corners is None or panel.rvec is None:
            continue

        # Position from the depth camera at the panel-center pixel.
        u, v = int(panel.center[0]), int(panel.center[1])
        depth = video_stream.get_depth_at(u, v)
        if depth is None:
            continue  # no valid depth (out of configured min/max range)
        # get_xyz_at already returns project axes (x-right, y-fwd, z-up), the
        # same convention PnP's tvec relabel produces, so position is directly
        # comparable downstream. Scale metres -> cm.
        xyz = video_stream.get_xyz_at(u, v, depth)
        position = np.array(xyz, dtype=np.float64) * METERS_TO_CM

        # Camera-relative panel yaw from the PnP rotation matrix (rvec is
        # axis*angle, so its z-component is not the yaw — extract via R).
        R, _ = cv.Rodrigues(panel.rvec)
        yaw_rad = -np.arctan2(R[0, 2], R[2, 2]) + np.pi
        yaw_rad = (yaw_rad + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
        if abs(np.degrees(yaw_rad)) < 90:
            display.windows["main"].add_contour(panel.inner_corners, color=CYAN)
            display.windows["main"].add_contour(panel.outer_corners, color=BLUE)
            panels.append(
                ArmorPanel(
                    icon=panel.id if hasattr(panel, "id") else None,
                    position=position,
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


class ClassicalDepthDetectorModule(Module[Context]):
    """Detects armor panels; position from a depth camera, orientation from PnP."""

    def __init__(self, context: Context):
        super().__init__(
            name="classical_depth_detector",
            context=context,
            inputs=[],
            outputs=["panels"],
        )

    @real()
    def _run_detect(self) -> Optional[List[ArmorPanel]]:
        """Process the current video frame and populate *ctx.panels*."""
        from src.subsystems.video_streaming.video_stream import video_stream

        f = video_stream.get_frame()
        frame = f.data
        self.ctx.frame_ts = f.timestamp
        display.windows["main"].img = frame
        assert frame is not None, "No frame received."

        contours = frame_proccesing.frame_process(frame)
        lights = armor.bounding_boxes(contours, frame)

        # Not enough lights to form a panel
        if len(lights) <= 1:
            return None

        try:
            pairs = armor.pairing(lights)
        except Exception as e:
            print(f"Error in pairing: {e}")
            return None

        return _process_pairs(pairs, frame)
