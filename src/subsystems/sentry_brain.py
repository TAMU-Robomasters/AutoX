"""The sentry behaviour brain — a pure ``py_trees`` tree (plan 09, Phase 4).

Doctrine (the behaviour the operator asked for): **progress to the point; if an
enemy is seen, move slowly side-to-side (evade) until no enemies remain; then keep
progressing to the point.** Turret auto-aim runs fail-open throughout — the brain
governs the *chassis*, not the gun (so ``engage`` is always true here; the field is
kept so a future doctrine can hold fire).

This module is deliberately **rclpy-free and side-effect-free**: it maps a small
:class:`BrainInputs` (what the world looks like) to a :class:`BrainOutputs` (what
the chassis should do). The :class:`~src.engines.autonav.AutoNav` engine feeds it
live inputs (enemy visibility from auto-aim, nav status) and turns its outputs into
``nav2`` ``NavigateToPose`` goals. Keeping it pure means the doctrine is unit-tested
on a laptop with no ROS, and can be lifted into any runtime.

Tree::

    Selector (root)
    ├── Sequence "engage_evade"        # higher priority when an enemy is visible
    │   ├── EnemyVisible?               # FAILURE when clear -> fall through
    │   └── SidestepEvade              # outputs.chassis = "evade"
    └── ProgressToPoint                # outputs.chassis = "progress" (default)
"""

from dataclasses import dataclass

import py_trees
from py_trees.common import Status

# Chassis modes the brain emits; the AutoNav engine maps each to nav2 motion.
CHASSIS_PROGRESS = "progress"  # NavigateToPose toward the current waypoint
CHASSIS_EVADE = "evade"  # slow side-to-side strafe (small lateral relative goals)


@dataclass
class BrainInputs:
    """What the brain knows about the world this tick."""

    enemy_visible: bool = False  # auto-aim currently has a live target


@dataclass
class BrainOutputs:
    """What the brain wants the chassis (and gun) to do this tick."""

    chassis: str = CHASSIS_PROGRESS  # CHASSIS_PROGRESS | CHASSIS_EVADE
    engage: bool = True  # let auto-aim fire (fail-open; this doctrine never holds)


class _EnemyVisible(py_trees.behaviour.Behaviour):
    """SUCCESS while an enemy is visible, else FAILURE (gates the evade branch)."""

    def __init__(self, brain: "SentryBrain"):
        super().__init__("EnemyVisible")
        self._brain = brain

    def update(self) -> Status:  # noqa: D102
        return Status.SUCCESS if self._brain.inputs.enemy_visible else Status.FAILURE


class _SetChassis(py_trees.behaviour.Behaviour):
    """Set the chassis mode on the brain's outputs and succeed."""

    def __init__(self, name: str, brain: "SentryBrain", mode: str):
        super().__init__(name)
        self._brain = brain
        self._mode = mode

    def update(self) -> Status:  # noqa: D102
        self._brain.outputs.chassis = self._mode
        return Status.SUCCESS


class SentryBrain:
    """Pure behaviour tree mapping :class:`BrainInputs` -> :class:`BrainOutputs`."""

    def __init__(self) -> None:
        self.inputs = BrainInputs()
        self.outputs = BrainOutputs()
        self._root = self._build_tree()
        self._tree = py_trees.trees.BehaviourTree(self._root)

    def _build_tree(self) -> py_trees.behaviour.Behaviour:
        engage_evade = py_trees.composites.Sequence("engage_evade", memory=False)
        engage_evade.add_children(
            [_EnemyVisible(self), _SetChassis("SidestepEvade", self, CHASSIS_EVADE)]
        )
        root = py_trees.composites.Selector("Sentry", memory=False)
        root.add_children(
            [engage_evade, _SetChassis("ProgressToPoint", self, CHASSIS_PROGRESS)]
        )
        return root

    def tick(self, inputs: BrainInputs) -> BrainOutputs:
        """Tick the tree once against ``inputs`` and return the decided outputs."""
        self.inputs = inputs
        self.outputs = BrainOutputs()  # fresh defaults; the tree overrides chassis
        self._tree.tick()
        return self.outputs
