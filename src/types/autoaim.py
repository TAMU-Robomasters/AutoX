"""Shared types for the auto-aim pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.core.module import Context
from src.toolbox.geometry_tools import BoundingBox


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass
class ArmorPanel:
    """Container for armor panel info from any robot.

    Attributes:
        icon: Detected icon/ID on the panel (int index from icon matching).
        position: 3-D translation vector (tvec) from PnP.
        orientation: 3-D rotation vector (rvec) from PnP.
        bbx: Panel bounding box in image coordinates.
        yaw: Panel yaw angle in radians (set after turret-frame transform).
    """

    icon: Optional[int]
    position: Optional[np.ndarray]
    orientation: Optional[np.ndarray]
    bbx: Optional[BoundingBox]
    contour: Optional[np.ndarray]
    yaw: float = 0.0


# ---------------------------------------------------------------------------
# Robot types
# ---------------------------------------------------------------------------


ICON_TO_ROBOT_NAME: Dict[int, str] = {
    0: "sentry",
    1: "hero",
    2: "standard",
    3: "sentry",
}
"""Mapping from icon detection index to robot name.

Icon indices depend on the sorted order of icon files:
  0 -> complete_sentry_icon.jpeg  -> sentry
  1 -> cropped_1.png              -> hero
  2 -> cropped_3.png              -> standard
  3 -> cropped_sentry.png         -> sentry
"""


@dataclass
class EnemyRobot:
    """An enemy robot identified by its name and the panels that belong to it."""

    name: str
    panels: List[ArmorPanel] = field(default_factory=list)


# ---------------------------------------------------------------------------
# State estimation and ballistic solution
# ---------------------------------------------------------------------------


@dataclass
class RobotStateEstimate:
    """State estimate produced by the particle filter.

    Attributes:
        value: 7-D state vector [x_c, y_c, vx, vy, theta, omega, radius].
        timestamp: ``time.perf_counter()`` at the moment the estimate was computed.
        confidence: Effective sample ratio (N_eff / N) from the particle filter.
    """

    value: np.ndarray
    timestamp: float
    confidence: float


@dataclass
class BallisticSolution:
    """Final firing solution sent to the embedded system.

    Attributes:
        pitch: Barrel pitch angle in radians.
        yaw: Barrel yaw angle in radians.
        alignment_time_ms: Milliseconds until the target panel faces us.
    """

    pitch: float
    yaw: float
    alignment_time_ms: int


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass
class AutoAimContext(Context):
    """Context passed through the simple auto-aim module pipeline."""

    timestamp: Optional[float] = None
    panels: Optional[List[ArmorPanel]] = None
    target_panel: Optional[ArmorPanel] = None
    prev_target_panel: Optional[ArmorPanel] = None
    radii: Optional[np.ndarray] = None


@dataclass
class ParticleFilterAutoAimContext(Context):
    """Context passed through the particle-filter auto-aim pipeline."""

    # Detection stage
    panels: Optional[List[ArmorPanel]] = None

    # Classification stage
    sentry: Optional[EnemyRobot] = None
    hero: Optional[EnemyRobot] = None
    standard: Optional[EnemyRobot] = None

    # Targeting stage
    target_robot: Optional[EnemyRobot] = None

    # Estimation stage
    estimate: Optional[RobotStateEstimate] = None

    # Ballistic stage
    solution: Optional[BallisticSolution] = None
