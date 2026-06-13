"""Unit tests for PanelHeightDeltaModule (signed inter-pair height learning)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.estimation.height_delta import PanelHeightDeltaModule  # noqa: E402
from src.types.autoaim import (  # noqa: E402
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
)

CENTER = (10.0, 240.0)
RADIUS = 23.5


def _panel_at(theta, k, z, panel_id):
    angle = theta + k * np.pi / 2
    return ArmorPanel(
        icon=None,
        position=np.array(
            [CENTER[0] + RADIUS * np.cos(angle), CENTER[1] + RADIUS * np.sin(angle), z],
            dtype=np.float64,
        ),
        orientation=None,
        bbx=None,
        contour=None,
        yaw=angle,
        id=panel_id,
    )


def _run_frames(module, ctx, frames):
    results = []
    for panels in frames:
        ctx.target_robot = EnemyRobot(name="standard", panels=panels)
        results.append(module.run().height_delta_estimate)
    return results


@pytest.mark.parametrize("z_even,z_odd", [(10.0, 4.0), (4.0, 10.0)])
def test_converges_to_signed_delta(z_even, z_odd):
    """Dz converges to z_even - z_odd with the correct sign, both directions."""
    rng = np.random.default_rng(1)
    ctx = FullStateAutoAimContext()
    module = PanelHeightDeltaModule(ctx)

    frames = []
    for i in range(40):
        even = _panel_at(0.1 * i, 0, z_even + rng.normal(0, 0.5), panel_id=0)
        odd = _panel_at(0.1 * i, 1, z_odd + rng.normal(0, 0.5), panel_id=1)
        frames.append([even, odd])

    results = _run_frames(module, ctx, frames)
    final = results[-1]
    assert final.dz == pytest.approx(z_even - z_odd, abs=0.5)
    assert final.var < results[0].var


def test_skips_non_adjacent_and_single_panel():
    """Same-parity ids, missing ids, or a single panel hold the filter state."""
    ctx = FullStateAutoAimContext()
    module = PanelHeightDeltaModule(ctx)
    _run_frames(module, ctx, [[_panel_at(0, 0, 10.0, 0), _panel_at(0, 1, 4.0, 1)]])

    frames = [
        [_panel_at(0, 0, 10.0, 0), _panel_at(0, 2, 4.0, 2)],   # same parity
        [_panel_at(0, 0, 10.0, None), _panel_at(0, 1, 4.0, None)],  # no ids
        [_panel_at(0, 0, 10.0, 0)],                              # single panel
    ]
    results = _run_frames(module, ctx, frames)
    assert all(r is not None and r.n_updates == 1 for r in results)


def test_no_target_returns_none():
    """No target robot: output stays None."""
    ctx = FullStateAutoAimContext(target_robot=None)
    module = PanelHeightDeltaModule(ctx)
    assert module.run().height_delta_estimate is None
