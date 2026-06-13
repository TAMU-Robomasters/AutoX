"""Unit tests for robot_constants: pair geometry, canonicalize, anchoring, persistence."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.subsystems.estimation.robot_constants import (  # noqa: E402
    PairMeasurement,
    RobotConstants,
    anchor_parity,
    canonicalize,
    load_constants,
    match_parity,
    measure_adjacent_pair,
    save_constants,
)
from src.toolbox.storage import JsonStore  # noqa: E402


def _panel(center, theta, radius, k, z):
    """Synthetic panel k of a robot at heading theta: (position, yaw)."""
    angle = theta + k * np.pi / 2
    pos = np.array(
        [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), z]
    )
    return pos, angle


# ---------------------------------------------------------------------------
# measure_adjacent_pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, 0.7, -2.1])
def test_measure_adjacent_pair_recovers_synthetic_radii(theta):
    """Two exact 90deg-apart panels yield their true radii and height delta."""
    center = (55.0, 210.0)
    pos_a, yaw_a = _panel(center, theta, radius=30.0, k=0, z=12.0)
    pos_b, yaw_b = _panel(center, theta, radius=20.0, k=1, z=7.0)

    m = measure_adjacent_pair(pos_a, yaw_a, pos_b, yaw_b)
    assert m is not None
    assert m.r_a == pytest.approx(30.0, abs=1e-9)
    assert m.r_b == pytest.approx(20.0, abs=1e-9)
    assert m.dz == pytest.approx(5.0)


def test_measure_adjacent_pair_order_independent():
    """Passing (b, a) instead of (a, b) measures the same geometry, swapped."""
    center = (0.0, 300.0)
    pos_a, yaw_a = _panel(center, 0.4, radius=27.0, k=2, z=0.0)
    pos_b, yaw_b = _panel(center, 0.4, radius=22.0, k=3, z=-3.0)

    m_ab = measure_adjacent_pair(pos_a, yaw_a, pos_b, yaw_b)
    m_ba = measure_adjacent_pair(pos_b, yaw_b, pos_a, yaw_a)
    assert m_ab is not None and m_ba is not None
    assert m_ab.r_a == pytest.approx(m_ba.r_b)
    assert m_ab.dz == pytest.approx(-m_ba.dz)


def test_measure_adjacent_pair_rejects_non_adjacent():
    """Opposite panels (180deg apart) and same-direction ghosts return None."""
    center = (0.0, 200.0)
    pos_a, yaw_a = _panel(center, 0.0, radius=25.0, k=0, z=0.0)
    pos_c, yaw_c = _panel(center, 0.0, radius=25.0, k=2, z=0.0)  # opposite panel
    assert measure_adjacent_pair(pos_a, yaw_a, pos_c, yaw_c) is None
    assert measure_adjacent_pair(pos_a, yaw_a, pos_a + 1.0, yaw_a) is None


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------


def test_canonicalize_even_pair_lower():
    """Dz < 0 means the even pair is lower; it becomes r_low."""
    rc = canonicalize(r_even=20.0, r_odd=27.0, dz=-6.0,
                      height_degenerate_cm=1.0, radii_degenerate_cm=1.0)
    assert rc.r_low == 20.0 and rc.r_high == 27.0
    assert rc.height_delta == 6.0
    assert rc.lower_is_minor is True
    assert not rc.degenerate_height and not rc.degenerate_radii


def test_canonicalize_odd_pair_lower_and_major():
    """Dz > 0: odd pair is lower; here the lower pair has the LARGER radius."""
    rc = canonicalize(r_even=20.0, r_odd=27.0, dz=6.0,
                      height_degenerate_cm=1.0, radii_degenerate_cm=1.0)
    assert rc.r_low == 27.0 and rc.r_high == 20.0
    assert rc.lower_is_minor is False


def test_canonicalize_degenerate_height_keys_by_radius():
    """Equal heights: stored keyed by radius (minor pair = 'low' by convention)."""
    rc = canonicalize(r_even=27.0, r_odd=20.0, dz=0.2,
                      height_degenerate_cm=1.0, radii_degenerate_cm=1.0)
    assert rc.degenerate_height and not rc.degenerate_radii
    assert rc.r_low == 20.0 and rc.r_high == 27.0
    assert rc.height_delta == 0.0 and rc.lower_is_minor is True


def test_canonicalize_full_degenerate_circle():
    """Circle at one height: mean radius, nothing to anchor."""
    rc = canonicalize(r_even=23.4, r_odd=23.6, dz=0.1,
                      height_degenerate_cm=1.0, radii_degenerate_cm=1.0)
    assert rc.degenerate_height and rc.degenerate_radii
    assert rc.r_low == rc.r_high == pytest.approx(23.5)


# ---------------------------------------------------------------------------
# anchor_parity
# ---------------------------------------------------------------------------


def test_anchor_parity_by_relative_height():
    """The lower observed parity gets r_low -- relative z, both orderings."""
    rc = RobotConstants(r_low=20.0, r_high=27.0, height_delta=6.0,
                        lower_is_minor=True, degenerate_height=False,
                        degenerate_radii=False)

    # Even-parity panel observed lower (works on a platform too: only relative).
    anchor = anchor_parity(rc, z_even=100.0, z_odd=106.0)
    assert anchor.low_parity == 0
    assert anchor.r_even == 20.0 and anchor.r_odd == 27.0

    # Odd-parity panel observed lower.
    anchor = anchor_parity(rc, z_even=106.0, z_odd=100.0)
    assert anchor.low_parity == 1
    assert anchor.r_even == 27.0 and anchor.r_odd == 20.0


def test_anchor_parity_degenerate_height_uses_measured_radii():
    """Equal heights: the smaller measured radius identifies the minor (low) pair."""
    rc = RobotConstants(r_low=20.0, r_high=27.0, height_delta=0.0,
                        lower_is_minor=True, degenerate_height=True,
                        degenerate_radii=False)

    m = PairMeasurement(r_a=26.5, r_b=20.5, dz=0.0)  # odd pair measured smaller
    anchor = anchor_parity(rc, z_even=100.0, z_odd=100.0, measured=m)
    assert anchor.low_parity == 1
    assert anchor.r_even == 27.0 and anchor.r_odd == 20.0


def test_anchor_parity_full_degenerate_is_arbitrary_but_consistent():
    """Circle: anchoring picks parity 0 and equal radii -- no crash."""
    rc = RobotConstants(r_low=23.5, r_high=23.5, height_delta=0.0,
                        lower_is_minor=True, degenerate_height=True,
                        degenerate_radii=True)
    anchor = anchor_parity(rc, z_even=100.0, z_odd=100.0)
    assert anchor.low_parity == 0
    assert anchor.r_even == anchor.r_odd == 23.5


# ---------------------------------------------------------------------------
# match_parity
# ---------------------------------------------------------------------------


def test_match_parity_detects_swap():
    """A measurement matching the swapped hypothesis returns True."""
    stored = np.array([20.0, 27.0, -6.0])
    assert match_parity(stored, np.array([20.5, 26.5, -5.8])) is False
    assert match_parity(stored, np.array([26.8, 20.2, 6.1])) is True


def test_match_parity_degenerate_prefers_identity():
    """Equidistant (fully degenerate) measurements default to identity."""
    stored = np.array([23.5, 23.5, 0.0])
    assert match_parity(stored, np.array([23.5, 23.5, 0.0])) is False


# ---------------------------------------------------------------------------
# persistence round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    """RobotConstants survive a JsonStore round trip; malformed entries skipped."""
    store = JsonStore(tmp_path / "constants.json")
    rc = RobotConstants(r_low=20.0, r_high=27.0, height_delta=6.0,
                        lower_is_minor=True, degenerate_height=False,
                        degenerate_radii=False)
    save_constants(store, "standard", rc)
    store.set("broken", {"r_low": 1.0})  # malformed: missing fields

    loaded = load_constants(JsonStore(tmp_path / "constants.json"))
    assert loaded == {"standard": rc}
