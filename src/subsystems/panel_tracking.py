"""Panel-ID tracking: label each visible panel with its index on the robot.

Contract (turret frame, cm, angles from +x CCW -- see src/types/autoaim.py):
    inputs:  ``estimate`` (previous frame's full-state estimate),
             ``target_robot``
    outputs: ``target_robot`` (same object; ``panel.id`` written in place)

Panel ``k`` of a robot at heading ``theta`` sits at ``center + r_k *
[cos(theta + k*90deg), sin(theta + k*90deg)]``. Each detected panel is matched
to the predicted direction it best agrees with (dot product of the normalized
center->panel vector against the four predicted directions), greedily and
uniquely.

INVARIANT (shared with kf.py and ballistics/solver.py): id ``k`` <=> angle
``theta + k*90deg``; only id *parity* (0&2 vs 1&3) is physically meaningful.
Ids are defined relative to the current theta track: they become meaningless
whenever the full-state estimator reinits (the engine must re-anchor).
``set_parity_offset`` is the single place parity flips, so every downstream
parity consumer (radii, height-delta, estimation) stays mutually consistent.

Latency note: this runs on the *previous* frame's ``estimate`` (the engine
orders tracking before estimation), one frame of staleness. Near the 45deg
match boundary or with dropped frames a misassignment is possible; downstream
measurement gates and KF smoothing bound the damage.
"""

from typing import Optional

import numpy as np

from src.core.module import Module, mock, real
from src.types.autoaim import (
    EnemyRobot,
    FullStateAutoAimContext,
    RobotStateEstimate,
)


class PanelTrackingModule(Module[FullStateAutoAimContext]):
    """Assign ids 0..3 to visible panels using the full-state estimate."""

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="panel_tracking",
            context=context,
            inputs=["estimate", "target_robot"],
            outputs=["target_robot"],
        )
        self._parity_offset: int = 0

    def set_parity_offset(self, offset: int) -> None:
        """Shift all assigned ids by ``offset`` (0 or 1) -- the parity re-anchor."""
        self._parity_offset = int(offset) % 4

    def reset(self) -> None:
        """Clear the parity offset (new theta track)."""
        self._parity_offset = 0

    @real()
    def _run_tracking(
        self,
        estimate: Optional[RobotStateEstimate],
        target_robot: Optional[EnemyRobot],
    ) -> Optional[EnemyRobot]:
        if target_robot is None:
            return None
        if estimate is None or not target_robot.panels:
            return target_robot  # pass through; ids stay None

        center = np.asarray(estimate.value[:2], dtype=float)
        theta = float(estimate.value[4])
        angles = theta + np.arange(4) * (np.pi / 2.0)
        directions = np.column_stack([np.cos(angles), np.sin(angles)])  # (4, 2)

        panels = [p for p in target_robot.panels if p.position is not None]
        scores = np.full((len(panels), 4), -np.inf)
        for i, panel in enumerate(panels):
            offset = np.asarray(panel.position, dtype=float).flatten()[:2] - center
            norm = np.linalg.norm(offset)
            if norm < 1e-6:
                continue  # panel at the estimated centre: direction undefined
            scores[i] = directions @ (offset / norm)

        # Greedy unique assignment: best score first, each panel/id used once.
        remaining_panels = set(range(len(panels)))
        remaining_ids = set(range(4))
        while remaining_panels and remaining_ids:
            i_raw, k_raw = np.unravel_index(int(np.argmax(scores)), scores.shape)
            i, k = int(i_raw), int(k_raw)
            if not np.isfinite(scores[i, k]):
                break  # only degenerate candidates left
            panels[i].id = (k + self._parity_offset) % 4
            scores[i, :] = -np.inf
            scores[:, k] = -np.inf
            remaining_panels.discard(i)
            remaining_ids.discard(k)

        return target_robot

    @mock
    def _run_mock_tracking(
        self,
        estimate: Optional[RobotStateEstimate],
        target_robot: Optional[EnemyRobot],
    ) -> Optional[EnemyRobot]:
        """Pass-through for mock mode."""
        return target_robot
