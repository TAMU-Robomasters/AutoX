"""Unit tests for the sentry behaviour brain (plan 09, Phase 4).

The brain is a pure ``py_trees`` tree, so its doctrine is exercised here with no
ROS / rclpy / hardware: progress to the point; on enemy contact slow-strafe to
evade until clear; then resume progressing. Auto-aim stays fail-open throughout.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.sentry_brain import (  # noqa: E402
    CHASSIS_EVADE,
    CHASSIS_PROGRESS,
    BrainInputs,
    SentryBrain,
)


def test_brain_is_rclpy_free():
    """Importing/using the brain must not pull in ROS (it's laptop-pure)."""
    SentryBrain().tick(BrainInputs(enemy_visible=False))
    assert "rclpy" not in sys.modules


def test_progress_when_no_enemy():
    """With no enemy visible the chassis progresses toward the point."""
    out = SentryBrain().tick(BrainInputs(enemy_visible=False))
    assert out.chassis == CHASSIS_PROGRESS
    assert out.engage is True  # auto-aim stays fail-open


def test_evade_when_enemy_visible():
    """An enemy in view switches the chassis to the slow side-step evade."""
    out = SentryBrain().tick(BrainInputs(enemy_visible=True))
    assert out.chassis == CHASSIS_EVADE
    assert out.engage is True  # still firing while evading


def test_resumes_progress_after_enemy_clears():
    """The full doctrine over time: progress -> evade on contact -> progress."""
    brain = SentryBrain()
    seq = [False, False, True, True, False]
    modes = [brain.tick(BrainInputs(enemy_visible=e)).chassis for e in seq]
    assert modes == [
        CHASSIS_PROGRESS,
        CHASSIS_PROGRESS,
        CHASSIS_EVADE,
        CHASSIS_EVADE,
        CHASSIS_PROGRESS,
    ]


def test_tick_is_stateless_across_instances():
    """Each tick reflects only the current input (no latched evade)."""
    brain = SentryBrain()
    assert brain.tick(BrainInputs(enemy_visible=True)).chassis == CHASSIS_EVADE
    # Enemy gone the very next tick -> immediately back to progressing.
    assert brain.tick(BrainInputs(enemy_visible=False)).chassis == CHASSIS_PROGRESS
