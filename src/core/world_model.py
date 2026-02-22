"""A singleton used to store information about the world (i.e. enemy positions) for all processes to access."""

from src.types.autoaim import AutoAimContext
from src.types.world_model import (  # noqa: F401 – re-exported
    CurrentWorldModel,
    EnemyRobot,
)


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
