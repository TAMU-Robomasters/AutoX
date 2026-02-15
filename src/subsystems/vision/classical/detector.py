"""Classic vision processing module for auto-aim pipeline.

Capable of finding position, orientation, and icon of armor panels
"""
from typing import List, Optional

import numpy as np

from src.core.module import Module
from src.subsystems.video_streaming.video_stream import video_stream
from src.subsystems.vision.classical import armor, frame_proccesing, icon_detection, pnp
from src.subsystems.vision.types import ArmorPanel
from src.subsystems.display import display
from src.toolbox.geometry_tools import BoundingBox


def _process_pairs(pairs, frame):
    panels = []
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
            panels.append(panel)
    return panels

def detect() -> Optional[List[ArmorPanel]]:
    """Process the video frame to detect armor panels."""
    frame = video_stream.get_frame()
    display.windows["main"].img = frame
    assert frame is not None, "No frame received from video stream."

    contours = frame_proccesing.frame_process(frame)
    lights = armor.bounding_boxes(contours, frame) 

    # Not enough lights to form a panel
    if len(lights) <= 1:
        cords = []
        return None

    try:
        pairs = armor.pairing(lights)
    except Exception as e:
        print(f"Error in pairing: {e}")
        cords = []
        return None
    panels = _process_pairs(pairs, frame)


    panels = []
    for i in range(len(panels)):
        panels.append(ArmorPanel(
            panels[0].id if len(panels) > 0 else None,
            panels[0].tvec if len(panels) > 0 else None,
            panels[0].rvec if len(panels) > 0 else None,
            BoundingBox.from_points(
                top_left=panels[i].corners[0][0],
                bottom_right=panels[i].corners[2][0]
            )
        ))
    
    return panels

