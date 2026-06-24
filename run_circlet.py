"""Standalone launcher for the circlet ring engine (headless, no auto-aim / MCU).

Spawns ONLY the ``CircletEngine`` and its four ``CameraDriver`` processes so the
ring of body-fixed cameras can be brought up and tested in isolation. The engine
detects per camera and publishes ``CircletDetections`` to a queue with no
subscriber (``publish`` is last-value / non-blocking), so this is a fully
self-contained test -- nothing consumes the detections.

Run it with the ``@CIRCLET_ONLY`` profile (real pyav capture on video0/2/4/6):

    uv run run_circlet.py @CIRCLET_ONLY

Watch the per-second ``circlet: capture fps [...]; published N panels/s`` log to
confirm all four cameras stream and the detector runs. Ctrl-C to stop.
"""

from src.core.orchestrator import launch_system
from src.engines.circlet import CircletEngine
from src.toolbox.logger import stop_log_listener


def main() -> None:
    """Launch the circlet engine + its camera drivers and join until interrupted."""
    processes = launch_system([CircletEngine])
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
