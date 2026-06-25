"""Evade integration test (plan 09): AutoNav + a fake auto-aim toggling aim_target.

Stands in for the full auto-aim engine (which needs a camera/detector) by
publishing ``aim_target`` on the cross-engine link AutoNav subscribes to: no enemy
for ENEMY_AFTER_S (the brain → progress to waypoints), then enemy visible (the
brain → slow side-step evade). Run against a live nav2 stack (rig 1):

    python run_evade_test.py autonav:nav2_localizer:none

Watch the BasicNavigator goals flip from waypoints (0.8,0)/(0,0) to lateral
side-steps (cx, cy±evade_offset) around the contact point, and the robot strafe.
"""

import time

from src.core.engine import Engine
from src.core.orchestrator import launch_system
from src.engines.autonav import AutoNav
from src.toolbox.logger import stop_log_listener
from src.types.null import NullContext
from src.types.sentry import AimTarget

ENEMY_AFTER_S = 25.0  # progress first, then "see an enemy" -> evade


class FakeAimEngine(Engine[NullContext]):
    """Publish aim_target=False, then True after ENEMY_AFTER_S (a stand-in enemy)."""

    publishes_queue = "aim_target"

    def __init__(self, driver_registry=None):
        """Construct with no modules (just publishes the fake target signal)."""
        super().__init__(
            modules=[], context_type=NullContext, driver_registry=driver_registry
        )

    def initialize(self):
        """Start the clock for the fake enemy schedule."""
        self._t0 = time.monotonic()
        self._last_log = -100.0

    def execute(self):
        """Publish the (scheduled) enemy visibility at ~5 Hz."""
        elapsed = time.monotonic() - self._t0
        visible = elapsed > ENEMY_AFTER_S
        self.publish(AimTarget(visible=visible))
        if elapsed - self._last_log > 3.0:
            self.log.info("fake aim: t=%4.0fs enemy_visible=%s", elapsed, visible)
            self._last_log = elapsed
        time.sleep(0.2)


def main():
    """Launch AutoNav + the fake auto-aim and join until interrupted."""
    processes = launch_system([AutoNav, FakeAimEngine])
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
    finally:
        stop_log_listener()


if __name__ == "__main__":
    main()
