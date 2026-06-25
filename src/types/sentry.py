"""Cross-engine messages for the sentry behaviour link (plan 09).

These travel over the interim ``multiprocessing.Queue`` pub/sub links between the
auto-aim engine and the AutoNav brain, so they must be plain picklable dataclasses
(no numpy / no ROS types). Both carry a wall-clock ``stamp`` so the consumer can
apply a freshness gate (a stale directive is ignored — auto-aim fails OPEN).
"""

from dataclasses import dataclass, field
from time import time


@dataclass
class AimTarget:
    """Auto-aim -> AutoNav: whether auto-aim currently has a live enemy target."""

    visible: bool
    stamp: float = field(default_factory=time)


@dataclass
class EngageDirective:
    """AutoNav -> auto-aim: whether auto-aim is cleared to engage.

    This doctrine always engages (the brain governs the chassis, not the gun), so
    ``engage`` is true; auto-aim treats a *missing or stale* directive as engage
    too (fail-OPEN), and only holds fire on a fresh ``engage=False``.
    """

    engage: bool
    stamp: float = field(default_factory=time)
