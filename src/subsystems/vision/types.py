"""Types for vision subsystems."""
import select

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from src.core.context import Context
from src.toolbox.geometry_tools import BoundingBox


@dataclass
class ArmorPanel: #TODO unify this dataclass with other similar ones
    """Container for armor panel info from any robot.
    
    Attributes:
        bbx: Panel bounding box 
    """
    icon: Optional[str]
    position: Optional[np.ndarray]
    orientation: Optional[np.ndarray]
    bbx: Optional[BoundingBox] 

@dataclass
class AutoAimContext(Context):
    timestamp: Optional[float] = None
    panels: Optional[List[ArmorPanel]] = None
    target_panel: Optional[ArmorPanel] = None
    prev_target_panel: Optional[ArmorPanel] = None