"""A singleton used to store information about the world (i.e. enemy positions) for all processes to access."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.pipelines.aim.context import AutoAimContext


# TODO: profile this to make sure it's not slow
@dataclass(frozen=True)  # make immutable
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


# TODO: make pipelines create their own update methods as multiprocessing safe modules
# TODO: make this multiprocessing safe
class WorldModel:
    """A singleton used to store information about the world."""

    def __init__(self):
        """Initialize the world model."""
        self._model: CurrentWorldModel = CurrentWorldModel(
            enemy_sentry=None, enemy_hero=None, enemy_standard=None
        )

    def update_with_aim(self, AimCtx: AutoAimContext):
        """Update the world model with information from the auto-aim pipeline."""
        pass

    def get_snapshot(self) -> CurrentWorldModel:
        """Get a snapshot of the current world model so no other process modifies it."""
        return self._model


world_model = WorldModel()
