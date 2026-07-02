#!/usr/bin/env python3
"""Live-tune ALL classical detector parameters against the real pipeline.

Runs the actual detector code -- ``frame_process`` -> ``bounding_boxes`` ->
``pairing`` -> ``armour_corners`` -> ``icon_detection`` from
``src/subsystems/vision/classical_detector/`` -- on live camera frames, with
every ``config.classical`` tunable on a trackbar. The values are written
straight into the loaded config each frame, so what you see is exactly what
the robot's detector would do with those numbers.

The preview shows the annotated camera frame (light boxes green, panel inner
corners cyan, panels green when the icon matched / yellow when it didn't)
beside the binary threshold mask. All windows are ``cv.WINDOW_NORMAL``
(freely resizable) and work over X11 forwarding (``ssh -X``) -- shrink them to
cut X11 bandwidth on a slow link. Camera exposure/controls from ``info.yaml``
are applied first so the preview matches the robot.

Trackbar notes:
    - ``x100`` sliders are the float value times 100 (e.g. 50 -> 0.5).
    - ``icon C +50`` is offset by +50 (trackbars can't go negative):
      slider 45 -> C = -5.
    - ``icon block`` is forced odd and >= 3 (adaptiveThreshold requirement).
    - For the pairing sliders, "stricter" (rejects more pairs) is: multipliers
      UP, thresholds DOWN (except ``hratio lo``: UP).

Usage (from the repo root; profile args pass through to quik-config):
    uv run python utils/tune_detector.py
    uv run python utils/tune_detector.py --device /dev/video0 @SENTRY
    uv run python utils/tune_detector.py --image path/to/frame.png

Keys (with the preview window focused):
    q / ESC  quit (prints the info.yaml snippet)
    e        export the snippet to utils/detector_settings.txt
    p        pause/resume on the current frame (sliders keep working)

If the camera is busy, the robot's autobooted AutoX probably owns it:
``sudo systemctl stop autox_boot.service`` first.

For tuning ONLY the color thresholds there is the simpler
``utils/tune_color_thresholds.py``; for camera exposure/gain/white-balance use
``utils/tune_camera.py``.
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

EXPORT_PATH = Path(__file__).resolve().parent / "detector_settings.txt"
CTRL_WINDOW = "detector tuning"
VIEW_WINDOW = "detector preview"
COLORS = ("red", "blue")

GREEN = (0, 255, 0)
CYAN = (255, 255, 137)
YELLOW = (85, 195, 254)

# (trackbar label, max). Positions are read every frame and written into
# config.classical by apply_sliders_to_config below.
SLIDERS = [
    ("0=red 1=blue", 1),
    ("thresh lo", 255),
    ("mask value hi", 255),
    ("angle mult x100", 1000),
    ("misalign mult x100", 1000),
    ("dist mult x100", 1000),
    ("hratio mult x100", 1000),
    ("angle thresh", 180),
    ("misalign thresh", 180),
    ("hratio lo x100", 300),
    ("hratio hi x100", 300),
    ("score thresh", 2000),
    ("icon block (odd)", 255),
    ("icon C +50", 100),
    ("icon tol", 255),
    ("need icon 0/1", 1),
]


def initial_positions(config, thresholds, color: str):
    """Seed every slider from the current config values."""
    cl = config.classical
    icon_block, icon_c = config_get(cl, "icon_adaptive_thresh", [101, -5])
    hr_lo, hr_hi = config_get(cl, "height_ratio_thresh", [0.5, 2.0])
    return {
        "0=red 1=blue": COLORS.index(color),
        "thresh lo": thresholds[color][0],
        "mask value hi": thresholds[color][1],
        "angle mult x100": int(round(cl.angle_diff_multiplier * 100)),
        "misalign mult x100": int(round(cl.misalignment_multiplier * 100)),
        "dist mult x100": int(round(cl.expected_distance_multiplier * 100)),
        "hratio mult x100": int(round(cl.height_ratio_multiplier * 100)),
        "angle thresh": int(cl.angle_diff_thresh),
        "misalign thresh": int(cl.misalignment_thresh),
        "hratio lo x100": int(round(hr_lo * 100)),
        "hratio hi x100": int(round(hr_hi * 100)),
        "score thresh": int(cl.score_thresh),
        "icon block (odd)": int(icon_block),
        "icon C +50": int(icon_c) + 50,
        "icon tol": int(config_get(cl, "icon_tolerance", 55)),
        "need icon 0/1": 1,
    }


def apply_sliders_to_config(config, thresholds, color: str) -> dict:
    """Read all trackbars and write the values into ``config.classical``.

    The detector functions read ``config.classical.*`` live on every call, so
    mutating the loaded config is all it takes for the real pipeline to run
    with the slider values. Returns the raw slider positions.
    """
    pos = {label: cv.getTrackbarPos(label, CTRL_WINDOW) for label, _hi in SLIDERS}
    cl = config.classical
    thresholds[color][0] = pos["thresh lo"]
    thresholds[color][1] = pos["mask value hi"]
    setattr(cl, f"{color}_thresh", list(thresholds[color]))
    cl.angle_diff_multiplier = pos["angle mult x100"] / 100
    cl.misalignment_multiplier = pos["misalign mult x100"] / 100
    cl.expected_distance_multiplier = pos["dist mult x100"] / 100
    cl.height_ratio_multiplier = pos["hratio mult x100"] / 100
    cl.angle_diff_thresh = pos["angle thresh"]
    cl.misalignment_thresh = pos["misalign thresh"]
    cl.height_ratio_thresh = [pos["hratio lo x100"] / 100, pos["hratio hi x100"] / 100]
    cl.score_thresh = pos["score thresh"]
    cl.icon_adaptive_thresh = [
        max(3, pos["icon block (odd)"]) | 1,
        pos["icon C +50"] - 50,
    ]
    cl.icon_tolerance = pos["icon tol"]
    return pos


def settings_text(config, thresholds) -> str:
    """The tuned values as a paste-ready info.yaml ``classical:`` snippet."""
    cl = config.classical
    hr = cl.height_ratio_thresh
    icon = cl.icon_adaptive_thresh
    lines = [
        f"# classical detector tuned {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# paste over the matching keys in the classical: block of src/info.yaml",
        f"blue_thresh: [ {thresholds['blue'][0]}, {thresholds['blue'][1]} ]",
        f"red_thresh: [ {thresholds['red'][0]}, {thresholds['red'][1]} ]",
        f"angle_diff_multiplier: {cl.angle_diff_multiplier:g}",
        f"misalignment_multiplier: {cl.misalignment_multiplier:g}",
        f"expected_distance_multiplier: {cl.expected_distance_multiplier:g}",
        f"height_ratio_multiplier: {cl.height_ratio_multiplier:g}",
        f"angle_diff_thresh: {cl.angle_diff_thresh}",
        f"misalignment_thresh: {cl.misalignment_thresh}",
        f"height_ratio_thresh: [ {hr[0]:g}, {hr[1]:g} ]",
        f"score_thresh: {cl.score_thresh}",
        f"icon_adaptive_thresh: [ {icon[0]}, {icon[1]} ]",
        f"icon_tolerance: {cl.icon_tolerance}",
    ]
    return "\n".join(lines) + "\n"


def make_mask(config, frame, color: str):
    """The mask frame_process builds, recomputed here for the side-by-side view."""
    channel = 2 if color == "red" else 0
    lo, hi = getattr(config.classical, f"{color}_thresh")
    _, thresh = cv.threshold(frame[:, :, channel], lo, hi, cv.THRESH_BINARY)
    return cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def main() -> int:
    """Parse args, open the camera, and run the detector tuning loop."""
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
    # armor.pairing must read config.classical directly (which we mutate), not
    # spawn its own in-engine trackbar window.
    config.classical.live_tuning = False
    config.classical.show_icon_filters = False

    # Import the real pipeline only after the config is loaded/adjusted.
    from src.subsystems.display import display
    from src.subsystems.vision.classical_detector import armor, frame_proccesing

    try:
        from src.subsystems.vision.classical_detector import icon_detection
    except Exception as e:  # icons folder missing/unreadable -- tune without icons
        icon_detection = None
        print(
            f"note: icon detection unavailable ({e}); panels shown without icon check"
        )

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

    thresholds = {
        c: [int(v) for v in config_get(config.classical, f"{c}_thresh", [200, 240])]
        for c in COLORS
    }
    color = config.our_team_color if config.our_team_color in COLORS else "red"

    cv.namedWindow(CTRL_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(CTRL_WINDOW, 520, 700)
    seeds = initial_positions(config, thresholds, color)
    for label, hi in SLIDERS:
        cv.createTrackbar(
            label, CTRL_WINDOW, max(0, min(hi, seeds[label])), hi, lambda _v: None
        )
    cv.namedWindow(VIEW_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(VIEW_WINDOW, 1280, 360)

    paused = False
    frame = still
    while True:
        new_color = COLORS[cv.getTrackbarPos("0=red 1=blue", CTRL_WINDOW)]
        if new_color != color:
            # Switching color: re-seed the threshold sliders from that color.
            color = new_color
            cv.setTrackbarPos("thresh lo", CTRL_WINDOW, thresholds[color][0])
            cv.setTrackbarPos("mask value hi", CTRL_WINDOW, thresholds[color][1])
        config.our_team_color = color
        pos = apply_sliders_to_config(config, thresholds, color)

        if still is None and not paused:
            frame = capture.latest()
        if frame is not None:
            # The real pipeline, exactly as ClassicalDetectorModule runs it.
            display.windows["main"].set_image(frame)
            contours = frame_proccesing.frame_process(frame)
            lights = armor.bounding_boxes(contours, frame)  # draws light boxes
            pairs = armor.pairing(lights) if len(lights) > 1 else []
            n_panels = 0
            img = display.windows["main"].img
            for pair in pairs:
                panel = armor.armour_corners(pair)
                icon_ok = icon_detection is not None and icon_detection.icon_detection(
                    panel, frame
                )
                if pos["need icon 0/1"] and icon_detection is not None and not icon_ok:
                    outline = YELLOW  # would be REJECTED by the icon check
                else:
                    n_panels += 1
                    outline = GREEN
                cv.drawContours(img, [panel.inner_corners], -1, CYAN, 2)
                cv.drawContours(img, [panel.corners], -1, outline, 2)
                if icon_ok and panel.id is not None:
                    cv.putText(
                        img,
                        f"icon {panel.id}",
                        (int(panel.center[0]), int(panel.center[1])),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        outline,
                        2,
                        cv.LINE_AA,
                    )

            hud = f"{color}  lights={len(lights)} pairs={len(pairs)} panels={n_panels}"
            hud += "  [PAUSED]" if paused else ""
            cv.putText(
                img,
                hud,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                YELLOW,
                2,
                cv.LINE_AA,
            )
            mask = make_mask(config, frame, color)
            view = np.hstack([img, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)])
            cv.imshow(VIEW_WINDOW, view)

        key = cv.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("e"):
            EXPORT_PATH.write_text(settings_text(config, thresholds))
            print(f"wrote {EXPORT_PATH}")

    if capture is not None:
        capture.stop()
    cv.destroyAllWindows()
    print(settings_text(config, thresholds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
