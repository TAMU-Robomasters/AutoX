"""Target selection module.

Picks which enemy robot to engage based on the closest panel distance.
"""

from typing import List, Optional

import numpy as np

from src.core.module import Module, real
from src.types.autoaim import (
    EnemyRobot,
    ParticleFilterAutoAimContext,
)


def _closest_panel_distance(robot: EnemyRobot) -> float:
    """Return the Euclidean distance of the robot's closest panel, or inf."""
    if not robot.panels:
        return float("inf")
    distances = [
        float(np.linalg.norm(p.position))
        for p in robot.panels
        if p.position is not None
    ]
    return min(distances) if distances else float("inf")


class TargetingModule(Module[ParticleFilterAutoAimContext]):
    """Select the closest robot as the target."""

    def __init__(self, context: ParticleFilterAutoAimContext):
        super().__init__(
            name="targeting",
            context=context,
            inputs=["sentry", "hero", "standard"],
            outputs=["target_robot"],
        )

    @real()
    def _run_targeting(
        self,
        sentry: Optional[EnemyRobot],
        hero: Optional[EnemyRobot],
        standard: Optional[EnemyRobot],
    ) -> Optional[EnemyRobot]:
        """Pick the robot whose closest panel has the smallest depth."""
        candidates: List[EnemyRobot] = []
        for robot in (sentry, hero, standard):
            if robot is not None and robot.panels:
                candidates.append(robot)

        if not candidates:
            print("[INFO] No valid targets found.")
            return None

        return min(candidates, key=_closest_panel_distance)
