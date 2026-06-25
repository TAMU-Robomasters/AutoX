"""Standalone launcher for the MCU<->ROS nav bridge (plan 09, rig 2).

Spawns ONLY the ``McuBridgeEngine`` and its shared ``McuDriver`` process so the
inverted-UART nav bridge can be brought up and tested in isolation: it polls the
MCU for odometry and publishes ROS ``/odom`` (+ TF), and turns ``/cmd_vel`` into
MCU velocity. With ``MCU=MOCK`` the driver's ``MockBackend`` integrates the last
commanded velocity, so publishing ``/cmd_vel`` actually moves the reported
``/odom`` — the nav loop closes on a laptop with no Gazebo.

    # in the autox container (ROS sourced, venv on PATH):
    python run_mcu_bridge.py @SENTRY ROS=NAV2 MCU=MOCK

Then from a ROS shell: ``ros2 topic echo /odom`` (watch it publish), and
``ros2 topic pub -r 5 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'``
(watch /odom advance). Ctrl-C to stop.
"""

from src.core.orchestrator import launch_system
from src.engines.mcu_bridge import McuBridgeEngine
from src.toolbox.logger import stop_log_listener


def main() -> None:
    """Launch the MCU bridge + its MCU driver and join until interrupted."""
    processes = launch_system([McuBridgeEngine])
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
