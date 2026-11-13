"""Target selection module.

Picks which enemy robot to engage based on the closest panel distance.
"""

from typing import List, Optional

from src.core.module import Module, real
from src.types.autoaim import (
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
)


def _closest_panel_distance(robot: EnemyRobot) -> float:
    """Return the Euclidean distance of the robot's closest panel, or inf."""
    return min((p.distance_cm() for p in robot.panels), default=float("inf"))


class TargetingModule(Module[FullStateAutoAimContext]):
    """Select the closest robot as the target."""

    def __init__(self, context: FullStateAutoAimContext):
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


class SingleTargetModule(Module[FullStateAutoAimContext]):
    """No robot identity (no icons): the (up to) 2 closest panels are the target.

    Used by the EKF engine with the no-icon ``LightPanelDetectorModule``; one
    robot shows at most 2 panels. Holding/losing the target is engine logic.
    """

    NAME = "target"

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="single_target",
            context=context,
            inputs=["panels"],
            outputs=["target_robot"],
        )

    @real()
    def _run_single_target(self, panels: Optional[List[ArmorPanel]]) -> Optional[EnemyRobot]:
        if not panels:
            return None
        closest = sorted(panels, key=lambda p: p.distance_cm())[:2]
        return EnemyRobot(name=self.NAME, panels=closest)
