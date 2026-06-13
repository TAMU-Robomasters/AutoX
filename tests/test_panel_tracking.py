"""Unit tests for PanelTrackingModule (panel-id assignment from the estimate)."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.panel_tracking import PanelTrackingModule  # noqa: E402
from src.types.autoaim import (  # noqa: E402
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
    RobotStateEstimate,
)


def _panel(x, y, z=0.0, yaw=0.0):
    return ArmorPanel(
        icon=None,
        position=np.array([x, y, z], dtype=np.float32),
        orientation=None,
        bbx=None,
        contour=None,
        yaw=yaw,
    )


def _estimate(cx, cy, theta):
    return RobotStateEstimate(
        value=np.array([cx, cy, 0, 0, theta, 0], dtype=np.float32),
        timestamp=0.0,
        confidence=1.0,
    )


def _run(estimate, panels):
    ctx = FullStateAutoAimContext(
        estimate=estimate,
        target_robot=EnemyRobot(name="standard", panels=panels) if panels is not None else None,
    )
    module = PanelTrackingModule(ctx)
    module.run()
    return module, ctx


def test_assigns_ids_by_predicted_direction():
    """Panels placed exactly at theta + k*90deg get id k, for two visible panels."""
    cx, cy, theta = 30.0, 250.0, 0.5
    r = 25.0
    panels = []
    for k in [1, 2]:
        angle = theta + k * np.pi / 2
        panels.append(_panel(cx + r * np.cos(angle), cy + r * np.sin(angle)))

    _run(_estimate(cx, cy, theta), panels)
    assert panels[0].id == 1
    assert panels[1].id == 2


def test_collision_resolved_uniquely():
    """Two panels nearest the same direction still receive distinct ids."""
    cx, cy, theta = 0.0, 200.0, 0.0
    r = 25.0
    # One panel exactly at k=0, another only 30deg away (also nearest k=0).
    p0 = _panel(cx + r * np.cos(theta), cy + r * np.sin(theta))
    p_close = _panel(
        cx + r * np.cos(theta + np.deg2rad(30)), cy + r * np.sin(theta + np.deg2rad(30))
    )

    _run(_estimate(cx, cy, theta), [p0, p_close])
    assert p0.id == 0  # exact match wins the greedy pick
    assert p_close.id == 1  # pushed to its second-best direction
    assert p0.id != p_close.id


def test_parity_offset_shifts_ids():
    """set_parity_offset(1) relabels every id by +1 mod 4."""
    cx, cy, theta = 0.0, 200.0, 0.2
    r = 25.0
    panels = []
    for k in [0, 1]:
        angle = theta + k * np.pi / 2
        panels.append(_panel(cx + r * np.cos(angle), cy + r * np.sin(angle)))

    ctx = FullStateAutoAimContext(
        estimate=_estimate(cx, cy, theta),
        target_robot=EnemyRobot(name="standard", panels=panels),
    )
    module = PanelTrackingModule(ctx)
    module.set_parity_offset(1)
    module.run()
    assert panels[0].id == 1
    assert panels[1].id == 2

    module.reset()
    module.run()
    assert panels[0].id == 0


def test_pass_through_without_estimate():
    """No estimate yet (first frame): panels flow through with ids untouched."""
    panels = [_panel(0.0, 200.0)]
    _, ctx = _run(None, panels)
    assert ctx.target_robot is not None
    assert panels[0].id is None


def test_no_target_returns_none():
    """No target robot: output stays None."""
    _, ctx = _run(_estimate(0, 200, 0), None)
    assert ctx.target_robot is None
