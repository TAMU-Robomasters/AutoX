"""Synthetic multi-camera scene for testing classification + targeting.

A *position-level* mock (no pixels, no PnP, no detector): ground-truth robots at
3-D positions are "seen" by several fake cameras that sample **asynchronously at
their own, runtime-variable fps** and emit :class:`ArmorPanel` detections
directly. This exercises ``RobotClassificationModule`` + ``TargetingModule``
under unsynchronized multi-camera input -- the decision-making half of what the
circlet engine will feed full-state auto-aim.

Coordinate convention matches ``src/types/autoaim.py`` (turret/chassis frame, cm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np

from src.types.autoaim import ArmorPanel

# Icon indices -> robot name come from ICON_TO_ROBOT_NAME (src/types/autoaim.py):
#   0 -> sentry, 1 -> hero, 2 -> standard, 3 -> sentry
ICON_SENTRY = 0
ICON_HERO = 1
ICON_STANDARD = 2
ICON_SENTRY_ALT = 3

# fps may be a constant or a function of time (runtime-variable rate).
FpsSpec = Union[float, Callable[[float], float]]


def _as_fps(spec: FpsSpec, t: float) -> float:
    return float(spec(t)) if callable(spec) else float(spec)


@dataclass
class GroundTruthRobot:
    """A robot in the scene: an icon plus a position trajectory (cm, turret frame)."""

    name: str
    icon: int
    position_at: Callable[[float], np.ndarray]  # t (s) -> (3,) cm

    def panel_at(
        self, t: float, rng: np.random.Generator, noise_cm: float
    ) -> ArmorPanel:
        """One noisy panel detection of this robot at time ``t``."""
        pos = np.asarray(self.position_at(t), dtype=float) + rng.normal(
            0.0, noise_cm, 3
        )
        return ArmorPanel(
            icon=self.icon,
            position=pos,
            orientation=None,
            bbx=None,
            contour=None,
            yaw=0.0,
        )


@dataclass
class SimScene:
    """A set of ground-truth robots that every camera can see."""

    robots: List[GroundTruthRobot]

    def panels_at(
        self, t: float, rng: np.random.Generator, noise_cm: float
    ) -> List[ArmorPanel]:
        """All robots' panels at time ``t`` (one per robot, with measurement noise)."""
        return [r.panel_at(t, rng, noise_cm) for r in self.robots]

    @staticmethod
    def at(name: str, icon: int, xyz) -> GroundTruthRobot:
        """Helper: a stationary robot at a fixed ``xyz`` (cm)."""
        pos = np.asarray(xyz, dtype=float)
        return GroundTruthRobot(name=name, icon=icon, position_at=lambda _t: pos)

    @staticmethod
    def moving(name: str, icon: int, start, end, duration: float) -> GroundTruthRobot:
        """Helper: a robot moving linearly from ``start`` to ``end`` over ``duration`` s."""
        a = np.asarray(start, dtype=float)
        b = np.asarray(end, dtype=float)
        return GroundTruthRobot(
            name=name,
            icon=icon,
            position_at=lambda t: a + (b - a) * min(max(t / duration, 0.0), 1.0),
        )


@dataclass
class FakeCameraStream:
    """One camera sampling the shared scene at its own (possibly variable) fps.

    ``sample(t)`` produces this camera's detections at time ``t`` and advances its
    internal schedule by ``1/fps``. ``due(t)`` says whether the camera is ready to
    produce a new sample yet -- the consumer polls all cameras on a fine clock so
    they fire independently (unsynchronized), each at its own rate.
    """

    scene: SimScene
    fps: FpsSpec
    name: str = "cam"
    phase: float = 0.0  # initial offset so cameras don't all fire on the same tick
    noise_cm: float = 0.5
    seed: int = 0
    _next_t: float = field(init=False, default=0.0)
    _latest: List[ArmorPanel] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._next_t = self.phase

    def due(self, t: float) -> bool:
        """True if this camera should emit a fresh sample at time ``t``."""
        return t >= self._next_t

    def sample(self, t: float) -> List[ArmorPanel]:
        """Emit detections at ``t`` and schedule the next sample by 1/fps."""
        self._latest = self.scene.panels_at(t, self._rng, self.noise_cm)
        self._next_t = t + 1.0 / max(_as_fps(self.fps, t), 1e-3)
        return self._latest

    @property
    def latest(self) -> List[ArmorPanel]:
        """The most recent detections this camera produced (empty before first sample)."""
        return self._latest


def merge_latest(streams: List[FakeCameraStream]) -> List[ArmorPanel]:
    """Concatenate the most recent detections across all cameras (what a consumer sees)."""
    merged: List[ArmorPanel] = []
    for s in streams:
        merged.extend(s.latest)
    return merged


def step_streams(
    streams: List[FakeCameraStream],
    duration: float,
    clock_hz: float = 1000.0,
    on_tick: Optional[Callable[[float, List[FakeCameraStream]], None]] = None,
) -> None:
    """Drive ``streams`` over ``duration`` s on a fine virtual clock.

    Each camera fires independently at its own fps (``due``/``sample``); after any
    camera updates on a tick, ``on_tick(t, streams)`` is invoked so the test can
    merge + run the pipeline against the current multi-camera view.
    """
    dt = 1.0 / clock_hz
    t = 0.0
    while t <= duration:
        updated = False
        for s in streams:
            if s.due(t):
                s.sample(t)
                updated = True
        if updated and on_tick is not None:
            on_tick(t, streams)
        t += dt
