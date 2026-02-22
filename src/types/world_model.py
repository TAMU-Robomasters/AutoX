"""Types representing the shared world model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EnemyRobot:
    """Enemy robot stats."""

    panel_position_1: Optional[np.ndarray]
    panel_orientation_1: Optional[np.ndarray]
    panel_position_2: Optional[np.ndarray]
    panel_orientation_2: Optional[np.ndarray]
    health: Optional[float]


@dataclass(frozen=True)
class CurrentWorldModel:
    """Model of the current world state."""

    enemy_sentry: Optional[EnemyRobot]
    enemy_hero: Optional[EnemyRobot]
    enemy_standard: Optional[EnemyRobot]
