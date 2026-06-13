"""Shared types for the auto-aim pipeline.

CANONICAL COORDINATE CONVENTION (turret/ballistic frame, after the MCU
transform): x -> right, y -> forward, z -> up, lengths in cm. All internal
angles (robot heading ``theta``, ``ArmorPanel.yaw``) are measured from +x,
counter-clockwise -- standard ``atan2(y, x)``. Panel ``k`` of a robot at
heading ``theta`` sits at ``center + r_k * [cos(theta + k*90deg),
sin(theta + k*90deg)]``, and that angle is also the panel's outward-normal
direction. The MCU yaw convention (0 at +y) exists only at the solver output
boundary (see ``src/subsystems/ballistics/solver.py``).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.core.module import Context
from src.toolbox.geometry_tools import BoundingBox


@dataclass
class Frame:
    """Container for a video frame and its associated metadata.

    ``seq`` is a monotonically increasing frame counter set by the camera
    producer; consumers use it to detect stale (already-processed) frames.
    For frames received over shared memory, ``data`` is a read-only zero-copy
    view whose backing sample is held alive on the instance (see FrameReader).
    """

    data: np.ndarray
    timestamp: float
    seq: int = 0


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass
class ArmorPanel:
    """Container for armor panel info from any robot.

    Attributes:
        icon: Detected icon/ID on the panel (int index from icon matching).
        position: 3-D translation vector (tvec) from PnP.
        orientation: 3-D rotation vector (rvec) from PnP.
        bbx: Panel bounding box in image coordinates.
        yaw: Panel yaw angle in radians (set after turret-frame transform);
            the panel's outward-normal angle in the canonical convention.
        id: Panel index 0..3 relative to the current theta track, assigned by
            PanelTrackingModule (panel id k sits at theta + k*90deg). Only the
            parity (0&2 vs 1&3) is physically meaningful; ids are invalidated
            whenever the full-state estimator reinits.
    """

    icon: Optional[int]
    position: Optional[np.ndarray]
    orientation: Optional[np.ndarray]
    bbx: Optional[BoundingBox]
    contour: Optional[np.ndarray]
    yaw: float = 0.0
    id: Optional[int] = None


# ---------------------------------------------------------------------------
# Robot types
# ---------------------------------------------------------------------------


ICON_TO_ROBOT_NAME: Dict[int, str] = {
    0: "sentry",
    1: "hero",
    2: "standard",
    3: "sentry",
}
"""Mapping from icon detection index to robot name.

Icon indices depend on the sorted order of icon files:
  0 -> complete_sentry_icon.jpeg  -> sentry
  1 -> cropped_1.png              -> hero
  2 -> cropped_3.png              -> standard
  3 -> cropped_sentry.png         -> sentry
"""


@dataclass
class EnemyRobot:
    """An enemy robot identified by its name and the panels that belong to it."""

    name: str
    #NOTE: maybe in the future have a property that's like sorted panels and sorts the panels by a list instead of
    # sorting inside the claassification module. that would make things more explicit
    panels: List[ArmorPanel] = field(default_factory=list)



# ---------------------------------------------------------------------------
# State estimation and ballistic solution
# ---------------------------------------------------------------------------


@dataclass
class RobotStateEstimate:
    """Full state estimate produced by the estimation module (PF or KF backend).

    Attributes:
        value: 7-D state vector [x_c, y_c, vx, vy, theta, omega, z] (the KF
            backend; the archived particle filter is 6-D). Index 6 ``z`` is the
            filtered panel-center height; consumers index 0,1,4,5 work for both.
        timestamp: ``time.perf_counter()`` at the moment the estimate was computed.
        confidence: Backend-specific quality (PF: N_eff/N; KF: exp(-NIS/2)).
        a_radius: Orbit radius of the even-parity panel pair (ids 0 & 2), cm.
        b_radius: Orbit radius of the odd-parity panel pair (ids 1 & 3), cm.
            Parity is relative to the current theta track (panel id k <=> panel
            angle theta + k*90deg); defaults are the legacy single-radius value.
        aim_z: Vertical aim height in the turret frame, cm -- the mid-height
            between the two panel pairs. For the KF backend this is the
            *filtered* z (state index 6, predicted on no-observation ticks);
            ``None`` until the engine anchors panel parity (state 1 / archived
            paths). The full-state ballistic modules require it.
    """

    value: np.ndarray
    timestamp: float
    confidence: float
    a_radius: float = 23.5
    b_radius: float = 23.5
    aim_z: Optional[float] = None


@dataclass
class PanelEstimate:
    """Single-panel xy track produced by SinglePanelEstimationModule.

    Attributes:
        value: 4-D state vector [x, y, vx, vy] of the tracked panel itself
            (not the robot centre), cm and cm/s, turret frame.
        z: Raw z of the tracked panel (cm, turret frame) -- the actual aim
            height; no config z_offset involved.
        panel_id: The tracked panel's id (None while ids are unassigned).
        timestamp: ``time.perf_counter()`` when the estimate was computed.
    """

    value: np.ndarray
    z: float
    panel_id: Optional[int]
    timestamp: float


@dataclass
class RadiiEstimate:
    """Per-parity panel-orbit radii produced by RadiiEstimatorModule.

    Keyed by panel-id parity (NOT sorted by size -- the pairing with panel ids
    is the whole point). Variances are the KF posterior diagonals, used by the
    engine's convergence gate.
    """

    r_even: float   # cm, radius of the id-parity-0 pair (ids 0 & 2)
    r_odd: float    # cm, radius of the id-parity-1 pair (ids 1 & 3)
    var_even: float
    var_odd: float
    n_updates: int


@dataclass
class HeightDeltaEstimate:
    """Signed inter-pair height difference produced by PanelHeightDeltaModule."""

    dz: float       # cm, z_even_pair - z_odd_pair (signed)
    var: float
    n_updates: int


@dataclass
class BallisticSolution:
    """Final firing solution sent to the embedded system.

    Attributes:
        pitch: Barrel pitch angle in radians.
        yaw: Barrel yaw angle in radians (MCU convention: 0 at +y).
        alignment_time_ms: Milliseconds until the target panel faces us.
        is_confident: ``False`` means "aim, hold fire" -- the solver could not
            produce a real firing solution (target out of range / no ballistic
            arc), and pitch/yaw are a straight-line tracking aim instead. The
            engine forwards pitch/yaw but reports ``CVState.NO_TARGET``.
    """

    pitch: float
    yaw: float
    alignment_time_ms: int
    is_confident: bool = True


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass
class AutoAimContext(Context):
    """Context passed through the simple auto-aim module."""

    timestamp: Optional[float] = None
        # Classification stage
    sentry: Optional[EnemyRobot] = None
    hero: Optional[EnemyRobot] = None
    standard: Optional[EnemyRobot] = None

    # Targeting stage
    target_robot: Optional[EnemyRobot] = None

    frame_ts: Optional[float] = None
    panels: Optional[List[ArmorPanel]] = None
    target_panel: Optional[ArmorPanel] = None
    prev_target_panel: Optional[ArmorPanel] = None
    radii: Optional[np.ndarray] = None


@dataclass
class FullStateAutoAimContext(Context):
    """Context passed through the full-state auto-aim pipeline.

    Each field is written by the engine or by exactly one module ("produced
    by"); modules declare which fields they read/write, and the engine
    validates that wiring at construction.
    """

    # -- engine-written (frame pacing / MCU transform), read by modules ------
    start_loop_time: Optional[float] = None
    frame: Optional[np.ndarray] = None            # produced by engine (FrameReader)
    frame_ts: Optional[float] = None              # produced by engine
    new_observation: bool = False                 # produced by engine (MCU transform ok)

    # -- vision ---------------------------------------------------------------
    panels: Optional[List[ArmorPanel]] = None     # produced by detection module

    # -- classification / targeting -------------------------------------------
    sentry: Optional[EnemyRobot] = None           # produced by classification
    hero: Optional[EnemyRobot] = None             # produced by classification
    standard: Optional[EnemyRobot] = None         # produced by classification
    target_robot: Optional[EnemyRobot] = None     # produced by targeting (+ panel_tracking ids)

    # -- estimation -------------------------------------------------------------
    estimate: Optional[RobotStateEstimate] = None               # produced by full-state estimation
    xy_estimate: Optional[PanelEstimate] = None                 # produced by single_panel_estimation
    radii_estimate: Optional[RadiiEstimate] = None              # produced by radii_estimator
    height_delta_estimate: Optional[HeightDeltaEstimate] = None  # produced by height_delta_estimator

    # -- ballistics ---------------------------------------------------------
    solution: Optional[BallisticSolution] = None  # produced by the active ballistic module
