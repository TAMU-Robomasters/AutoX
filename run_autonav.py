"""Standalone launcher for the AutoNav sentry brain (plan 09, rig 1).

Spawns ONLY the ``AutoNav`` engine. It drives nav2 via ``BasicNavigator``, so the
full nav2 stack must already be up (rig 1 = the Gazebo bringup); ``initialize``
blocks on ``waitUntilNav2Active`` until it is. With no auto-aim running, the
``aim_target`` link is never fed, so ``enemy_visible`` stays False and the brain
just patrols the configured waypoints (the evade branch is exercised once auto-aim
publishes ``aim_target`` — plan 09 Phase 5).

    # in the autox container, with rig 1 (Gazebo + nav2) up:
    python run_autonav.py @SENTRY ROS=NAV2

Ctrl-C to stop.
"""

from src.core.orchestrator import launch_system
from src.engines.autonav import AutoNav
from src.toolbox.logger import stop_log_listener


def main() -> None:
    """Launch the AutoNav engine and join until interrupted."""
    processes = launch_system([AutoNav])
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
