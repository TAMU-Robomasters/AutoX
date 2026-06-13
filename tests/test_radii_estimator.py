"""Unit tests for RadiiEstimatorModule (per-parity orbit radii learning)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.estimation.radii import RadiiEstimatorModule  # noqa: E402
from src.types.autoaim import (  # noqa: E402
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
)

R_EVEN, R_ODD = 30.0, 20.0
CENTER = (40.0, 260.0)


def _panel_at(center, theta, radius, k, z=0.0, panel_id=None):
    angle = theta + k * np.pi / 2
    return ArmorPanel(
        icon=None,
        position=np.array(
            [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), z],
            dtype=np.float64,
        ),
        orientation=None,
        bbx=None,
        contour=None,
        yaw=angle,
        id=panel_id,
    )


def _pair(theta, r_even=R_EVEN, r_odd=R_ODD, ids=(0, 1), noise=0.0, rng=None):
    """Two adjacent panels of a synthetic robot, optionally with position noise."""
    panels = []
    for k, r in zip(ids, [r_even if ids[0] % 2 == 0 else r_odd,
                          r_odd if ids[0] % 2 == 0 else r_even]):
        p = _panel_at(CENTER, theta, r, k, panel_id=k)
        if noise and rng is not None:
            p.position[:2] += rng.normal(0, noise, size=2)
        panels.append(p)
    return panels


def _run_frames(module, ctx, frames):
    results = []
    for panels in frames:
        ctx.target_robot = EnemyRobot(name="standard", panels=panels)
        results.append(module.run().radii_estimate)
    return results


def test_single_exact_measurement_recovers_radii_by_parity():
    """One exact two-panel frame seeds [r_even, r_odd] -- keyed by parity, unsorted."""
    ctx = FullStateAutoAimContext()
    module = RadiiEstimatorModule(ctx)
    (result,) = _run_frames(module, ctx, [_pair(theta=0.7)])

    assert result is not None
    assert result.r_even == pytest.approx(R_EVEN, abs=1e-6)
    assert result.r_odd == pytest.approx(R_ODD, abs=1e-6)
    assert result.n_updates == 1


def test_parity_preserved_when_even_is_smaller():
    """Regression vs the old major/minor sort: r_even < r_odd must survive."""
    ctx = FullStateAutoAimContext()
    module = RadiiEstimatorModule(ctx)
    (result,) = _run_frames(
        module, ctx, [_pair(theta=0.3, r_even=18.0, r_odd=28.0)]
    )
    assert result.r_even == pytest.approx(18.0, abs=1e-6)
    assert result.r_odd == pytest.approx(28.0, abs=1e-6)


def test_noisy_sequence_converges_and_variance_drops():
    """Noisy measurements converge toward truth with monotonically usable variance."""
    rng = np.random.default_rng(0)
    ctx = FullStateAutoAimContext()
    module = RadiiEstimatorModule(ctx)

    frames = [_pair(theta=0.1 * i, noise=1.0, rng=rng) for i in range(60)]
    results = _run_frames(module, ctx, frames)

    final = results[-1]
    assert final.n_updates >= 55  # the gate may drop a couple of outliers
    assert final.r_even == pytest.approx(R_EVEN, abs=1.0)
    assert final.r_odd == pytest.approx(R_ODD, abs=1.0)
    assert final.var_even < results[0].var_even
    assert final.var_even < 0.5


def test_non_adjacent_and_single_panel_hold_state():
    """Ids 0&2 (same parity), missing ids, or one panel: no update, state held."""
    ctx = FullStateAutoAimContext()
    module = RadiiEstimatorModule(ctx)
    _run_frames(module, ctx, [_pair(theta=0.0)])  # seed

    opposite = [
        _panel_at(CENTER, 0.0, R_EVEN, 0, panel_id=0),
        _panel_at(CENTER, 0.0, R_EVEN, 2, panel_id=2),
    ]
    no_ids = [
        _panel_at(CENTER, 0.0, R_EVEN, 0, panel_id=None),
        _panel_at(CENTER, 0.0, R_ODD, 1, panel_id=None),
    ]
    single = [_panel_at(CENTER, 0.0, R_EVEN, 0, panel_id=0)]

    results = _run_frames(module, ctx, [opposite, no_ids, single])
    assert all(r is not None and r.n_updates == 1 for r in results)


def test_outlier_measurement_gated():
    """A wildly inconsistent pair (id misassignment) is rejected by the gate."""
    ctx = FullStateAutoAimContext()
    module = RadiiEstimatorModule(ctx)
    _run_frames(module, ctx, [_pair(theta=0.0) for _ in range(10)])

    # Swap the radii (classic parity misassignment): a 10cm jump on each axis.
    bad = _pair(theta=0.0, r_even=R_ODD, r_odd=R_EVEN)
    (result,) = _run_frames(module, ctx, [bad])
    assert result.n_updates == 10  # not incorporated
    assert result.r_even == pytest.approx(R_EVEN, abs=0.5)


def test_per_robot_isolation_and_reset():
    """Filters are keyed by robot name; reset() clears one robot's state."""
    ctx = FullStateAutoAimContext()
    module = RadiiEstimatorModule(ctx)

    ctx.target_robot = EnemyRobot(name="standard", panels=_pair(theta=0.0))
    module.run()
    ctx.target_robot = EnemyRobot(
        name="hero", panels=_pair(theta=0.0, r_even=35.0, r_odd=25.0)
    )
    hero_result = module.run().radii_estimate
    assert hero_result.r_even == pytest.approx(35.0, abs=1e-6)

    module.reset("standard")
    ctx.target_robot = EnemyRobot(name="standard", panels=[])
    assert module.run().radii_estimate is None  # state gone, no measurement
