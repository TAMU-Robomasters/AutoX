"""Robot classification module.

Takes detected armor panels and groups them into three enemy robot types
(sentry, hero, standard) based on their icon IDs.
"""

from typing import List, Optional, Tuple

from src.core.module import Module, real
from src.types.autoaim import (
    ArmorPanel,
    EnemyRobot,
    ICON_TO_ROBOT_NAME,
    ParticleFilterAutoAimContext,
)


class RobotClassificationModule(Module[ParticleFilterAutoAimContext]):
    """Classify detected panels into sentry, hero, and standard robots."""

    def __init__(self, context: ParticleFilterAutoAimContext):
        super().__init__(
            name="robot_classification",
            context=context,
            inputs=["panels"],
            outputs=["sentry", "hero", "standard"],
        )

    @real()
    def _run_classify(
        self, panels: Optional[List[ArmorPanel]]
    ) -> Tuple[Optional[EnemyRobot], Optional[EnemyRobot], Optional[EnemyRobot]]:
        """Bucket panels into three robots by their icon ID."""

        # Create empty robots
        sentry = EnemyRobot(name="sentry")
        hero = EnemyRobot(name="hero")
        standard = EnemyRobot(name="standard")

        if not panels:
            return sentry, hero, standard

        for panel in panels:
            if panel.icon is None:
                continue
            robot_name = ICON_TO_ROBOT_NAME.get(int(panel.icon))
            if robot_name == "sentry":
                sentry.panels.append(panel)
            elif robot_name == "hero":
                hero.panels.append(panel)
            elif robot_name == "standard":
                standard.panels.append(panel)

        return sentry, hero, standard
