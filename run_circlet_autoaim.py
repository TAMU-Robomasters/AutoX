"""Combined fps test: CircletCaptureEngine (4 ring cams, NO detect) + autoaim.

Launches the capture-only circlet ring alongside the full
``FullStateAutoAimEngine`` (gimbal cam detection + targeting), sharing one
``launch_system`` so all five cameras + the MCU driver come up together. Use it
to measure, under the real combined USB/IPC load:

  * ``circlet: capture fps [...]``      -- the 4 ring cameras' raw frame rate
  * ``fps: obs=.. no_obs=..``           -- the autoaim gimbal-cam loop rate

Run with a robot profile (configures the gimbal cam) + circlet enable, e.g.::

    uv run run_circlet_autoaim.py @SENTRY @CIRCLET_ONLY

Ctrl-C to stop.
"""

from src.core.orchestrator import launch_system
from src.engines.circlet import CircletCaptureEngine
from src.engines.full_state_autoaim import FullStateAutoAimEngine
from src.toolbox.logger import stop_log_listener


def main() -> None:
    """Launch capture-only circlet + autoaim and join until interrupted."""
    processes = launch_system([CircletCaptureEngine, FullStateAutoAimEngine])
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
