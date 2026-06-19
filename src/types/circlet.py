"""Cross-engine message types for the circlet ring (CircletEngine -> auto-aim).

These cross the process boundary over a pickled ``multiprocessing.Queue`` (the
``plans/03`` interim shortcut), so they are plain picklable dataclasses -- the
heavy/variable-length detector fields (``contour``, ``bbx``) are dropped; only
what the consumer needs for classification + target selection is carried.

Positions are in the **chassis frame** (cm): the circlet cameras are body-fixed,
so each camera applies only its static camera->chassis extrinsic before
publishing. The auto-aim engine lifts chassis -> turret with the live gimbal
yaw/pitch when it merges (see ``src/subsystems/circlet_support.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CircletPanel:
    """One armor panel seen by a circlet camera, in the chassis frame.

    Attributes:
        icon: Detected icon/ID (maps to a robot via ``ICON_TO_ROBOT_NAME``).
        position: (3,) chassis-frame translation, cm (x right / y forward / z up).
        yaw: Outward-normal azimuth in the chassis frame, radians (atan2(y, x)).
        camera_id: Which ring camera produced it (0..N-1).
        timestamp: ``time.perf_counter()`` of the source frame.
    """

    icon: Optional[int]
    position: np.ndarray
    yaw: float
    camera_id: int
    timestamp: float


@dataclass
class CircletDetections:
    """All circlet panels published in one tick (across all ring cameras)."""

    panels: List[CircletPanel] = field(default_factory=list)
    timestamp: float = 0.0
