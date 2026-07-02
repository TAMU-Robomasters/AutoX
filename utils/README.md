# Tuning tools

Three standalone GUI tools for tuning the vision stack, all designed to run
over **X11 forwarding** (`ssh -X orin@<robot>`) — no wifi or browser needed.
Every window is freely resizable and the video scales to fit, so shrink the
window if the link is slow.

| Tool | What it tunes | Export file |
| --- | --- | --- |
| `tune_camera.py` | V4L2 camera controls (exposure, gain, white balance, ...) | `camera_settings.txt` |
| `tune_color_thresholds.py` | `red_thresh` / `blue_thresh` (the detector's color masks) | `color_threshold_settings.txt` |
| `tune_detector.py` | **Every** `classical:` parameter, running the real detector pipeline | `detector_settings.txt` |

All three are run the same way, from the repo root:

```sh
uv run python utils/tune_camera.py
uv run python utils/tune_color_thresholds.py
uv run python utils/tune_detector.py
```

## Before you start

- **X11**: connect with `ssh -X` (or `ssh -Y`). The tools exit with a clear
  error if `$DISPLAY` is not set. With a monitor attached, `export DISPLAY=:0`.
- **Free the camera**: on the robot, the autobooted AutoX instance owns the
  camera. Stop it first:

  ```sh
  sudo systemctl stop autox_boot.service
  ```

  (Re-enable by rebooting, or `sudo systemctl start autox_boot.service`.)
- **Profiles**: quik-config `@PROFILE` args pass straight through, e.g.
  `uv run python utils/tune_detector.py @SENTRY`. Without one, the profiles
  persisted in `src/local_data.ignore.yaml` apply (on the robots that is the
  robot's own profile, so usually you type nothing).

## `tune_camera.py` — camera controls

Discovers whatever V4L2 controls the camera exposes (`v4l2-ctl
--list-ctrls-menus`) and builds an input box / checkbox / dropdown for each,
applied live while a preview runs. This is the same mechanism the camera
driver uses at startup, so the values you find are exactly what the robot
applies.

- Type a value and press Enter (or click away) to apply it.
- The preview scales to the window — resize freely.
- **Export settings** writes `utils/camera_settings.txt` containing a ready
  `v4l2-ctl` command plus the `info.yaml` fields; the same text is printed
  when you quit.
- Where the values go: `hardware.camera_exposure` and
  `hardware.camera_controls` in the robot's profile in `src/info.yaml`.

```sh
uv run python utils/tune_camera.py --device /dev/video0 --width 1280 --height 720 --fps 90
uv run python utils/tune_camera.py --no-preview   # controls only, no video
```

## `tune_color_thresholds.py` — red/blue masks

Reproduces exactly what the classical detector's `frame_process` does
(threshold one BGR channel, 3×3 morphological close, find contours) with the
two `classical.{red,blue}_thresh` values on sliders. The preview shows the
camera frame with found contours beside the binary mask the detector actually
sees. Tune until the enemy light bars are solid white and everything else is
black.

Note: the second value of each pair is `cv.threshold`'s *maxval* (the value
written into the mask), **not** an upper cutoff — the first number is the one
that matters for detection.

## `tune_detector.py` — the whole classical detector

Runs the **actual** pipeline code (`frame_process` → `bounding_boxes` →
`pairing` → `armour_corners` → `icon_detection`) on live frames, with every
`classical:` tunable on a trackbar: color thresholds, the four pairing score
multipliers, the pairing valid-mask thresholds, and the icon-detection
parameters. Slider values are written into the loaded config each frame, so
what you see is exactly what the robot would do with those numbers.

Reading the preview:

- green rotated boxes = detected light bars
- cyan = panel inner (light-bar) corners, what PnP would use
- green panel outline = kept; yellow = would be rejected by the icon check
- right half = the binary threshold mask
- HUD line = current color + lights/pairs/panels counts

Slider conventions (trackbars are integer and can't go negative):

- `x100` sliders are the float value ×100 (slider 50 → 0.5)
- `icon C +50` is offset by +50 (slider 45 → C = −5)
- `icon block` is forced odd and ≥ 3
- for the pairing sliders, "stricter" (rejects more pairs) is: multipliers
  **up**, thresholds **down** (except `hratio lo`: up)

Related: with an engine already running you can instead set
`classical.live_tuning: true` in `info.yaml` to get the in-engine pairing
sliders (`src/subsystems/vision/classical_detector/tuning.py`) — but that
covers only the pairing parameters; this tool covers everything and needs no
engine.

## Keys and common flags (both detector tools)

| Key | Action |
| --- | --- |
| `q` / `ESC` | quit and print the `info.yaml` snippet |
| `e` | export the snippet to the tool's `utils/*_settings.txt` |
| `p` | pause/resume on the current frame (sliders keep working) |

Flags: `--device /dev/videoN`, `--width/--height/--fps` (defaults come from
config), `--image path.png` to tune on a still image (no camera needed — works
while the autoboot instance owns the camera), `--no-camera-controls` to skip
pushing the config'd exposure/controls to the camera first.

## Saving your results

Changes are **ephemeral** — nothing is written to `info.yaml` automatically.
When you're happy, export (`e`) or quit (`q`) and paste the printed snippet
over the matching keys in the `classical:` block (or `hardware:` block for
camera settings) of `src/info.yaml`, in the right robot profile.

A sensible full-tuning order: `tune_camera.py` (exposure/gain first — the
masks depend on it) → `tune_color_thresholds.py` → `tune_detector.py`.

## Other files in `utils/`

- `tuning_common.py` — shared plumbing for the two detector tools (config
  loading, camera capture, v4l2 control push). Not a script.
- `camera_calibration/` — camera intrinsics generation (see `CLAUDE.md`).
- `autox_boot.sh`, `setup_boot_script.sh`, `kill_onboot_cv`, `start` —
  autoboot service management.
- `push_img_file_to_sd_card`, `save_img_file_from_sd_card` — SD-card image
  transfer helpers.
- `*_settings.txt` — the tuning tools' exported snippets.
