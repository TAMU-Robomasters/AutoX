"""Shared types for the auto-aim pipeline."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.core.module import Context
from src.toolbox.geometry_tools import BoundingBox


@dataclass
class ArmorPanel:
    """Container for armor panel info from any robot.

    Attributes:
        icon: Detected icon/ID on the panel.
        position: 3-D translation vector (tvec) from PnP.
        orientation: 3-D rotation vector (rvec) from PnP.
        bbx: Panel bounding box in image coordinates.
    """

    icon: Optional[str]
    position: Optional[np.ndarray]
    orientation: Optional[np.ndarray]
    bbx: Optional[BoundingBox]

@dataclass
class AutoAimContext(Context):
    """Context passed through the auto-aim module pipeline."""

    timestamp: Optional[float] = None
    panels: Optional[List[ArmorPanel]] = None
    target_panel: Optional[ArmorPanel] = None
    prev_target_panel: Optional[ArmorPanel] = None
    
    # Additional fields for panel duration and interval estimation
    panel_duration: Optional[float] = None
    rotation_interval: Optional[float] = None
