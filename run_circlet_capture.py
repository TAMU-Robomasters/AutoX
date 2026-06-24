"""Standalone CAPTURE-ONLY camera tester (headless, NO detection).

Reads frames zero-copy through the real ``CameraDriver`` + IPC ring but runs NO
detector, so the per-second ``circlet: capture fps [...]`` log is the true
per-camera frame throughput -- the USB/bandwidth picture isolated from detection
cost. (fps counts only genuinely new frames: the driver advances a seq per real
capture and the worker counts only when seq changes, so it can't inflate from
stale frames.)

It opens either FOUR or FIVE cameras depending on the profile:

  * ``uv run run_circlet_capture.py @CIRCLET_ONLY``
        -- the four circlet RING cameras only (no gimbal-cam config needed).

  * ``uv run run_circlet_capture.py @SENTRY @CIRCLET_ONLY``
        -- ALSO opens the main/gimbal cam (the ``frames`` driver). Needs a robot
        profile to supply ``config.hardware`` camera fields; the gimbal cam is
        addressed by ``hardware.camera_index`` (give it the stable
        ``/dev/circlet/maincam`` symlink, not an int, so it survives /dev/videoN
        renumbering).

Selection is automatic: if the gimbal cam is configured (``hardware.camera_width``
present) it opens all five, otherwise just the ring. Ctrl-C to stop.
"""

from src.core.orchestrator import launch_system
from src.engines.circlet import CircletCaptureAllEngine, CircletCaptureEngine
from src.toolbox.globals import config
from src.toolbox.logger import get_logger, stop_log_listener

log = get_logger("run_circlet_capture")


def main() -> None:
    """Launch the capture engine (4 ring cams, or 5 incl. gimbal cam) until ^C."""
    # The gimbal cam (frames driver) only has config under a robot profile; only
    # open it when that config is present, so @CIRCLET_ONLY alone still works.
    gimbal_configured = getattr(config.hardware, "camera_width", None) is not None
    engine = CircletCaptureAllEngine if gimbal_configured else CircletCaptureEngine
    log.info(
        "capture test: %s (%s)",
        engine.__name__,
        "4 ring cams + gimbal" if gimbal_configured else "4 ring cams only",
    )

    processes = launch_system([engine])
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
