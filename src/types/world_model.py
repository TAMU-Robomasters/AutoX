"""Types representing the shared world model."""

from dataclasses import dataclass
from typing import Optional

from src.types.autoaim import EnemyRobot


@dataclass(frozen=True)
class CurrentWorldModel:
    """Model of the current world state."""

    enemy_sentry: Optional[EnemyRobot]
    enemy_hero: Optional[EnemyRobot]
    enemy_standard: Optional[EnemyRobot]
