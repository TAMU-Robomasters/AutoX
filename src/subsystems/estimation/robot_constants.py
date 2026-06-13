"""Learned per-enemy-robot geometry constants: dataclass + pure helper functions.

The auto-aim engine learns each enemy robot's panel-orbit radii and the height
difference between its two panel pairs (state 1), saves them here in a
*canonical, track-independent* form, and re-anchors them onto live panel ids
when it next sees the robot (state 2a). Everything in this file is pure
(no context, no modules, no disk side effects beyond the two thin JsonStore
adapters), so it is unit-testable in isolation.

Canonical form: keyed by the LOWER pair (``r_low``/``r_high``,
``height_delta = |z_high - z_low| >= 0``), because "which pair is lower" is
observable in any future session by comparing two live panel z values --
panel-id parity is only meaningful within one theta track and cannot be saved.

Coordinate convention: see ``src/types/autoaim.py`` (turret frame, cm, angles
from +x CCW).
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from src.toolbox.logger import get_logger
from src.toolbox.storage import JsonStore
from src.types.autoaim import ArmorPanel

log = get_logger("robot_constants")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RobotConstants:
    """Canonical learned geometry of one enemy robot (all lengths in cm)."""

    r_low: float            # orbit radius of the LOWER panel pair
    r_high: float           # orbit radius of the higher pair
    height_delta: float     # |z_high - z_low| >= 0
    lower_is_minor: bool    # the lower pair has the smaller radius
    degenerate_height: bool  # height_delta below threshold -> anchor by radii instead
    degenerate_radii: bool   # radii ~equal (circle) -> anchoring is irrelevant

    def to_dict(self) -> dict:
        """JSON-serializable form for JsonStore."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RobotConstants":
        """Inverse of :meth:`to_dict` (raises TypeError/KeyError on malformed data)."""
        return cls(
            r_low=float(data["r_low"]),
            r_high=float(data["r_high"]),
            height_delta=float(data["height_delta"]),
            lower_is_minor=bool(data["lower_is_minor"]),
            degenerate_height=bool(data["degenerate_height"]),
            degenerate_radii=bool(data["degenerate_radii"]),
        )


@dataclass
class PairMeasurement:
    """One-shot geometry measured from two adjacent (90deg-apart) panels."""

    r_a: float   # cm, orbit radius of panel a
    r_b: float   # cm, orbit radius of panel b
    dz: float    # cm, z_a - z_b (signed)


@dataclass
class ParityAnchor:
    """Result of mapping saved constants onto live panel-id parity (state 2a)."""

    r_even: float        # radius to use for even-parity ids (0 & 2)
    r_odd: float         # radius for odd-parity ids (1 & 3)
    low_parity: int      # 0 or 1: which parity is the LOW pair (aim_z sign)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def measure_adjacent_pair(
    pos_a: np.ndarray,
    yaw_a: float,
    pos_b: np.ndarray,
    yaw_b: float,
    yaw_tol: float = np.radians(20.0),
) -> Optional[PairMeasurement]:
    """Measure both orbit radii + height delta from two adjacent panels.

    Panels of a 4-panel robot are 90deg apart, so their inward unit vectors
    form an exact orthonormal basis; the spin centre decomposes as
    ``c = (p_b . u_a) u_a + (p_a . u_b) u_b`` (section 2.1 of the
    state-estimation doc). Inputs are turret-frame positions (cm) and panel
    yaws (outward-normal angles from +x). Returns None if the panels' yaw
    separation is not ~90deg (non-adjacent ids or detector ghosts).
    """
    d_yaw = _wrap_pi(yaw_b - yaw_a)
    if abs(abs(d_yaw) - np.pi / 2) > yaw_tol:
        return None

    u_a = -np.array([np.cos(yaw_a), np.sin(yaw_a)])  # inward: panel -> centre
    # u_b = u_a rotated by exactly +-90deg (sign from the measured yaw order),
    # keeping the basis exactly orthogonal regardless of per-panel yaw noise.
    if d_yaw > 0:
        u_b = np.array([-u_a[1], u_a[0]])
    else:
        u_b = np.array([u_a[1], -u_a[0]])

    p_a = np.asarray(pos_a, dtype=float).flatten()
    p_b = np.asarray(pos_b, dtype=float).flatten()
    c = (p_b[:2] @ u_a) * u_a + (p_a[:2] @ u_b) * u_b

    return PairMeasurement(
        r_a=float(np.linalg.norm(c - p_a[:2])),
        r_b=float(np.linalg.norm(c - p_b[:2])),
        dz=float(p_a[2] - p_b[2]),
    )


def measure_pair_by_parity(panels: List[ArmorPanel]) -> Optional[PairMeasurement]:
    """Parity-keyed pair measurement from a robot's visible panels.

    Returns a :class:`PairMeasurement` with ``r_a`` = the even-parity panel's
    radius, ``r_b`` = the odd-parity panel's, and ``dz = z_even - z_odd`` --
    the exact form the radii / height-delta KFs and ``match_parity`` consume.
    Returns None unless there are exactly two panels, both with positions and
    tracked ids of *adjacent* parity, ~90deg apart (the geometric gate).
    """
    if len(panels) != 2:
        return None
    a, b = panels
    if a.position is None or b.position is None or a.id is None or b.id is None:
        return None
    if (a.id - b.id) % 2 != 1:  # same parity (e.g. ids 0 & 2): not adjacent
        return None
    if a.id % 2 == 1:
        a, b = b, a  # a := even-parity panel
    assert a.position is not None and b.position is not None  # checked above
    return measure_adjacent_pair(a.position, a.yaw, b.position, b.yaw)


# ---------------------------------------------------------------------------
# Canonicalization / anchoring
# ---------------------------------------------------------------------------


def canonicalize(
    r_even: float,
    r_odd: float,
    dz: float,
    height_degenerate_cm: float,
    radii_degenerate_cm: float,
) -> RobotConstants:
    """Convert converged per-parity estimates into the canonical save form.

    Args:
        r_even: Converged radius of the even-parity pair (cm).
        r_odd: Converged radius of the odd-parity pair (cm).
        dz: Converged signed height delta ``z_even - z_odd`` (cm).
        height_degenerate_cm: |dz| below this is treated as "equal heights".
        radii_degenerate_cm: |r_even - r_odd| below this is treated as a circle.
    """
    degenerate_height = abs(dz) < height_degenerate_cm
    degenerate_radii = abs(r_even - r_odd) < radii_degenerate_cm

    if not degenerate_height:
        if dz < 0:  # even pair is lower
            r_low, r_high = r_even, r_odd
        else:
            r_low, r_high = r_odd, r_even
        return RobotConstants(
            r_low=r_low,
            r_high=r_high,
            height_delta=abs(dz),
            lower_is_minor=r_low < r_high,
            degenerate_height=False,
            degenerate_radii=degenerate_radii,
        )

    if not degenerate_radii:
        # Heights equal: key by radius instead; "lower" is the minor pair by
        # convention (the aim_z sign is moot with height_delta = 0).
        return RobotConstants(
            r_low=min(r_even, r_odd),
            r_high=max(r_even, r_odd),
            height_delta=0.0,
            lower_is_minor=True,
            degenerate_height=True,
            degenerate_radii=False,
        )

    # Circle at one height: nothing to anchor.
    mean_r = (r_even + r_odd) / 2.0
    return RobotConstants(
        r_low=mean_r,
        r_high=mean_r,
        height_delta=0.0,
        lower_is_minor=True,
        degenerate_height=True,
        degenerate_radii=True,
    )


def anchor_parity(
    constants: RobotConstants,
    z_even: float,
    z_odd: float,
    measured: Optional[PairMeasurement] = None,
) -> ParityAnchor:
    """Map saved constants onto live panel-id parity (state 2a -> 2b).

    Called once two adjacent panels are visible and the engine has assigned
    them ids (one even-parity, one odd-parity). The *relative* z comparison
    decides which observed parity is the low pair -- never absolute z, so a
    robot standing on a platform anchors correctly.

    Args:
        constants: Saved canonical constants for this robot.
        z_even: Live z (cm) of the even-parity panel.
        z_odd: Live z (cm) of the odd-parity panel.
        measured: One-shot pair measurement with ``r_a`` = even-parity radius
            and ``r_b`` = odd-parity radius; used only when the saved height
            delta is degenerate.
    """
    if not constants.degenerate_height:
        low_parity = 0 if z_even < z_odd else 1
    elif not constants.degenerate_radii and measured is not None:
        # Heights equal: the pair with the smaller measured radius is the
        # minor pair, which canonicalize() stored as the "low" pair.
        low_parity = 0 if measured.r_a < measured.r_b else 1
    else:
        low_parity = 0  # circle (or no measurement): arbitrary, radii equal

    if low_parity == 0:
        return ParityAnchor(
            r_even=constants.r_low, r_odd=constants.r_high, low_parity=0
        )
    return ParityAnchor(r_even=constants.r_high, r_odd=constants.r_low, low_parity=1)


def match_parity(
    stored: np.ndarray,
    measured: np.ndarray,
) -> bool:
    """Decide whether a fresh pair measurement is parity-swapped vs stored state.

    Used in state 1 after target loss: panel ids restart with the new theta
    track, so the radii/height KF state (kept across the loss) may be
    parity-swapped relative to the new ids. Compares the raw measurement
    ``[r_even, r_odd, dz]`` against the stored KF means under identity vs swap
    (``[r_odd, r_even, -dz]``) and returns True if the swap fits better.

    When both radii and height are near-degenerate the two hypotheses are
    equidistant and identity (False) wins -- harmless, the quantities are equal.
    """
    stored = np.asarray(stored, dtype=float)
    measured = np.asarray(measured, dtype=float)
    swapped = np.array([stored[1], stored[0], -stored[2]])
    return bool(
        np.linalg.norm(measured - swapped) < np.linalg.norm(measured - stored)
    )


# ---------------------------------------------------------------------------
# Persistence adapters (thin; the engine owns the JsonStore instance)
# ---------------------------------------------------------------------------


def save_constants(store: JsonStore, name: str, constants: RobotConstants) -> None:
    """Persist one robot's constants (atomic write via JsonStore)."""
    store.set(name, constants.to_dict())


def load_constants(store: JsonStore) -> Dict[str, RobotConstants]:
    """Load all robots' constants, skipping malformed entries with a warning."""
    constants: Dict[str, RobotConstants] = {}
    for name in store.keys():
        try:
            constants[name] = RobotConstants.from_dict(store.get(name))
        except (KeyError, TypeError, ValueError):
            log.warning("ignoring malformed robot_constants entry %r", name)
    return constants
