from typing import List, Optional

import numpy as np

from src.core.module import Module, real
from src.subsystems.display import display
from src.types.autoaim import ArmorPanel, AutoAimContext


# Assume that you're only getting adjacent panels 
class RadiiEstimatorModule(Module[AutoAimContext]):
    """Estimates the radii of a robot."""

    def __init__(self, context: AutoAimContext):
        super().__init__(
            name="Radii Estimator",
            context=context,
            inputs=["panels"],
            outputs=["radii"],
        )

    @real()
    def _run_detect(self, panels: Optional[List[ArmorPanel]]) -> Optional[np.ndarray]:
        
        # LOGIC

        radii = np.array([0.0, 0.0])
        return radii

