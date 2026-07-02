#!/usr/bin/env python3
"""Shared plumbing for the ``utils/tune_*.py`` tuning tools.

Provides:

- :func:`load_autox_config` -- import AutoX's quik-config ``config`` without it
  eating the tool's own argparse flags.
- :func:`apply_camera_controls` -- push the config'd v4l2 controls (exposure +
  ``hardware.camera_controls``) to the camera, mirroring what the real camera
  driver does at startup, so the tuning preview matches what the robot sees.
- :class:`Capture` -- background OpenCV V4L2 MJPG capture thread holding only
  the newest frame.
- :func:`require_display` -- fail early with an ``ssh -X`` hint when there is
  no X11 display to open windows on.

Not a script; imported by ``tune_color_thresholds.py`` and ``tune_detector.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def require_display() -> None:
    """Exit with an X11-forwarding hint if there is no display to draw on."""
    if not os.environ.get("DISPLAY"):
        print(
            "error: $DISPLAY is not set -- no X11 display to open windows on.\n"
            "Over ssh, reconnect with `ssh -X` (or `ssh -Y`) to forward X11,\n"
            "or `export DISPLAY=:0` when a monitor is attached to the Jetson.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_autox_config(passthrough_args: Optional[List[str]] = None):
    """Import and return AutoX's quik-config ``config`` object.

    quik-config parses ``sys.argv`` at import time (the ``@PROFILE`` /
    ``key=value`` syntax), which collides with these tools' argparse flags. Call
    this AFTER ``argparse.parse_known_args()``, passing the leftover unparsed
    args: the tool's own flags are stripped from ``sys.argv`` and anything
    quik-config-shaped (e.g. ``@SENTRY``) is passed through. The profiles
    persisted in ``src/local_data.ignore.yaml`` still apply as usual.
    """
    sys.path.insert(0, str(REPO_ROOT))
    sys.argv = [sys.argv[0]] + list(passthrough_args or [])
    from src.toolbox.globals import config

    return config


def config_get(section, name: str, default=None):
    """``getattr`` that also treats a LazyDict's None-for-missing as missing."""
    value = getattr(section, name, None)
    return default if value is None else value


def default_device(config) -> str:
    """The /dev/video* path implied by ``hardware.camera_index`` (default 0)."""
    return f"/dev/video{config_get(config.hardware, 'camera_index', 0)}"


def default_geometry(config) -> Tuple[int, int, int]:
    """(width, height, fps) from ``hardware.camera_*``, else ``classical.cam_*``."""
    hw, cl = config.hardware, config.classical
    width = int(config_get(hw, "camera_width", config_get(cl, "cam_width", 1280)))
    height = int(config_get(hw, "camera_height", config_get(cl, "cam_height", 720)))
    fps = int(config_get(hw, "camera_fps", config_get(cl, "cam_fps", 90)))
    return width, height, fps


def apply_camera_controls(config, device: str) -> None:
    """Apply the same v4l2 controls the camera driver applies at startup.

    Mirrors ``CameraSource`` in ``src/drivers/video_stream.py``:
    ``hardware.camera_exposure`` expands to manual exposure (``auto_exposure=1``
    + ``exposure_time_absolute``), then everything in
    ``hardware.camera_controls`` is applied verbatim. Unsupported controls are
    skipped with a note, matching the driver's behavior.
    """
    controls = {}
    exposure = config_get(config.hardware, "camera_exposure")
    if exposure is not None:
        controls["auto_exposure"] = 1  # 1 = Manual Mode
        controls["exposure_time_absolute"] = int(exposure)
    controls.update(dict(config_get(config.hardware, "camera_controls") or {}))
    for name, value in controls.items():
        out = subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            print(f"note: skipped v4l2 control {name}={value}: {out.stderr.strip()}")


class Capture:
    """Background camera reader (OpenCV V4L2, MJPG) holding only the newest frame.

    The robot captures via ffmpeg, but for a tuning preview OpenCV's decoder is
    plenty and much simpler. If opening fails (typically because the autoboot
    AutoX instance owns the camera -- ``sudo systemctl stop autox_boot.service``)
    ``error`` is set instead of frames appearing.
    """

    def __init__(self, device: str, width: int, height: int, fps: int):
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> "Capture":
        """Spawn the background capture thread; returns self for chaining."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        import cv2 as cv

        idx = int(re.sub(r"\D", "", self.device) or 0)
        cap = cv.VideoCapture(idx, cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            self.error = (
                f"could not open {self.device} -- is another process using it? "
                "(if the robot autostarted AutoX, run: "
                "sudo systemctl stop autox_boot.service)"
            )
            return
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame
        cap.release()

    def latest(self):
        """Return the most recently captured BGR frame, or None."""
        with self._lock:
            return self._latest

    def wait_for_frame(self, timeout: float = 5.0):
        """Block until the first frame arrives (or *timeout*); returns it or None."""
        deadline = time.time() + timeout
        while time.time() < deadline and self.error is None:
            frame = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    def stop(self) -> None:
        """Signal the capture thread to stop and join it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
