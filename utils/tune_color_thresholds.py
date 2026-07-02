#!/usr/bin/env python3
"""Live-tune the classical detector's red/blue color thresholds.

Reproduces exactly what ``frame_process`` in
``src/subsystems/vision/classical_detector/frame_proccesing.py`` does -- take
the blue (or red) channel, ``cv.threshold`` it, close it with a 3x3 kernel,
find contours -- with the two ``config.classical.{blue,red}_thresh`` values on
trackbars. The preview shows the camera frame (found light contours drawn in
green) beside the binary mask the detector actually sees, so you can adjust
until the light bars are solid white and everything else is black.

Note on the two values: they are passed to ``cv.threshold(channel, lo, hi,
THRESH_BINARY)``, so ``lo`` is the cutoff (pixels above it count as light) and
``hi`` is just the value written into the mask -- ``lo`` is the one that
matters for detection.

Windows are ``cv.WINDOW_NORMAL`` (freely resizable) and the whole thing runs
fine over X11 forwarding (``ssh -X``) -- shrink the preview window to cut X11
bandwidth on a slow link. Camera exposure/controls from ``info.yaml`` are
applied first so the preview matches what the robot sees.

Usage (from the repo root; profile args pass through to quik-config):
    uv run python utils/tune_color_thresholds.py
    uv run python utils/tune_color_thresholds.py --device /dev/video0 @SENTRY
    uv run python utils/tune_color_thresholds.py --image path/to/frame.png

Keys (with a preview window focused):
    q / ESC  quit (prints the info.yaml snippet)
    e        export the snippet to utils/color_threshold_settings.txt
    p        pause/resume on the current frame

If the camera is busy, the robot's autobooted AutoX probably owns it:
``sudo systemctl stop autox_boot.service`` first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tuning_common import (
    Capture,
    apply_camera_controls,
    config_get,
    default_device,
    default_geometry,
    load_autox_config,
    require_display,
)

EXPORT_PATH = Path(__file__).resolve().parent / "color_threshold_settings.txt"
CTRL_WINDOW = "color threshold tuning"
VIEW_WINDOW = "color threshold preview"
COLORS = ("red", "blue")
CHANNEL = {"red": 2, "blue": 0}  # BGR channel index, as in frame_process


def make_mask(frame, color: str, lo: int, hi: int):
    """The detector's exact mask: threshold one channel, then 3x3 close."""
    _, thresh = cv.threshold(frame[:, :, CHANNEL[color]], lo, hi, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)


def settings_text(thresholds) -> str:
    """The tuned values as a paste-ready info.yaml snippet."""
    lines = [
        f"# color thresholds tuned {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# paste over the matching keys in the classical: block of src/info.yaml",
        f"blue_thresh: [ {thresholds['blue'][0]}, {thresholds['blue'][1]} ]",
        f"red_thresh: [ {thresholds['red'][0]}, {thresholds['red'][1]} ]",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Parse args, open the camera, and run the threshold tuning loop."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", help="V4L2 device (default: from config)")
    ap.add_argument("--width", type=int, help="capture width (default: from config)")
    ap.add_argument("--height", type=int, help="capture height (default: from config)")
    ap.add_argument("--fps", type=int, help="capture fps (default: from config)")
    ap.add_argument("--image", help="tune on a still image instead of the camera")
    ap.add_argument(
        "--no-camera-controls",
        action="store_true",
        help="don't apply info.yaml exposure/camera_controls to the camera first",
    )
    args, passthrough = ap.parse_known_args()

    require_display()
    config = load_autox_config(passthrough)

    still = None
    capture = None
    if args.image:
        still = cv.imread(args.image)
        if still is None:
            print(f"error: could not read image {args.image}", file=sys.stderr)
            return 1
    else:
        device = args.device or default_device(config)
        width, height, fps = default_geometry(config)
        if not args.no_camera_controls:
            apply_camera_controls(config, device)
        capture = Capture(
            device, args.width or width, args.height or height, args.fps or fps
        ).start()
        if capture.wait_for_frame() is None:
            print(f"error: {capture.error or 'no frames arrived'}", file=sys.stderr)
            return 1

    # Per-color [lo, hi], seeded from config so you start from current values.
    thresholds = {
        c: [int(v) for v in config_get(config.classical, f"{c}_thresh", [200, 240])]
        for c in COLORS
    }
    color = config.our_team_color if config.our_team_color in COLORS else "red"

    cv.namedWindow(CTRL_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(CTRL_WINDOW, 480, 130)
    cv.createTrackbar(
        "0=red 1=blue", CTRL_WINDOW, COLORS.index(color), 1, lambda _v: None
    )
    cv.createTrackbar(
        "thresh lo", CTRL_WINDOW, thresholds[color][0], 255, lambda _v: None
    )
    cv.createTrackbar(
        "mask value hi", CTRL_WINDOW, thresholds[color][1], 255, lambda _v: None
    )
    cv.namedWindow(VIEW_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(VIEW_WINDOW, 1280, 360)

    paused = False
    frame = still
    while True:
        new_color = COLORS[cv.getTrackbarPos("0=red 1=blue", CTRL_WINDOW)]
        if new_color != color:
            # Switching color: re-seed the sliders from that color's values.
            color = new_color
            cv.setTrackbarPos("thresh lo", CTRL_WINDOW, thresholds[color][0])
            cv.setTrackbarPos("mask value hi", CTRL_WINDOW, thresholds[color][1])
        thresholds[color][0] = cv.getTrackbarPos("thresh lo", CTRL_WINDOW)
        thresholds[color][1] = cv.getTrackbarPos("mask value hi", CTRL_WINDOW)

        if still is None and not paused:
            frame = capture.latest()
        if frame is not None:
            lo, hi = thresholds[color]
            mask = make_mask(frame, color, lo, hi)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            annotated = frame.copy()
            cv.drawContours(annotated, contours, -1, (0, 255, 0), 2)
            hud = f"{color}  lo={lo} hi={hi}  contours={len(contours)}"
            hud += "  [PAUSED]" if paused else ""
            cv.putText(
                annotated,
                hud,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (85, 195, 254),
                2,
                cv.LINE_AA,
            )
            view = np.hstack([annotated, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)])
            cv.imshow(VIEW_WINDOW, view)

        key = cv.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("e"):
            EXPORT_PATH.write_text(settings_text(thresholds))
            print(f"wrote {EXPORT_PATH}")

    if capture is not None:
        capture.stop()
    cv.destroyAllWindows()
    print(settings_text(thresholds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
