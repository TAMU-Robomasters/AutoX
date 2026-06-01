"""Calibrate a camera from chessboard images and emit artifacts for all three
runtime lens-correction modes.

You point this script at a directory of raw chessboard images (anywhere on disk —
they don't need to live in the repo) and give it a *name*. It creates
``assets/intrinsics/<name>/`` and writes everything the runtime needs there, so
the only thing you ever change in info.yaml is::

    camera_intrinsics_path: "<name>"
    lens_correction_mode: warp | unproject | normal

The three modes (config.hardware.lens_correction_mode), all served from the same
folder:

    warp       Remap the whole raw frame onto a pinhole image (mapxy.npy) before
               detection. PnP then runs on a distortion-free pinhole image.
    unproject  Detect on the raw frame; undistort only the PnP corner points
               through the splined model. Cheaper than warp (no full-frame
               remap) with the same accuracy on the corners.
    normal     Detect on the raw frame; feed the OpenCV rational model
               (LENSMODEL_OPENCV12) straight into cv.solvePnP, which handles the
               distortion natively. No mrcal math at runtime.

Pipeline:
    1. mrgingham detects the chessboard corners → corners.vnl
    2. mrcal-calibrate-cameras runs twice off that one corners cache:
         - the splined-stereographic model  (warp / unproject)
         - LENSMODEL_OPENCV12                (normal)
    3. a pinhole model is derived from the splined model for warp/unproject
    4. cv.calibrateCamera (rational model) is run as an independent cross-check
       of the mrcal OPENCV12 solve and printed in a side-by-side table

mrcal.pinhole_model_for_reprojection always puts the destination pinhole's
principal point at the geometric image center; the ``fit`` config field only
controls how the focal length / output dims are chosen:

    centers-horizontal  (default) — output dims close to source dims; minor
                                    vertical cropping.
    centers-vertical               — symmetric option; minor horizontal cropping.
    corners                        — maximize FOV; output dims grow to contain
                                    all four source corners (larger frame).

Artifacts written into assets/intrinsics/<name>/ (overwriting any previous run):
    corners.vnl           mrgingham corner detections (kept for re-solving)
    splined.cameramodel   splined-stereographic model      (unproject)
    opencv12.cameramodel  LENSMODEL_OPENCV12 model         (normal)
    pinhole.cameramodel   pinhole model derived from splined (warp; comparisons)
    mapxy.npy             (H, W, 2) remap for mrcal.transform_image (warp)
    camera_matrix.pkl     3x3 pinhole K, pickled numpy array (warp / unproject)
    dist.pkl              zeros, shape (1, 5)               (warp / unproject)

Calibration parameters (chessboard geometry, lens model, etc.) come from
calib_config.yaml next to this script. At runtime the script asks whether to use
that file as-is or prompt you for each value.

Install mrcal + mrgingham (not on PyPI, system packages only):
    sudo apt install python3-mrcal mrgingham
The uv-managed venv needs system site-packages access to see mrcal
(create with ``uv venv --python /usr/bin/python3.10 --system-site-packages``).

Usage:
    uv run python utils/camera_calibration/mrcal_calibrate_camera.py
    uv run python utils/camera_calibration/mrcal_calibrate_camera.py \
        --images ~/calib_shots --name mrcal_sentry_1920 --prompt-each
"""
import argparse
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2 as cv
import mrcal
import numpy as np
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
INTRINSICS_ROOT = REPO_ROOT / "assets/intrinsics"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "calib_config.yaml"

# mrcal LENSMODEL_OPENCV12 intrinsics layout = [fx, fy, cx, cy] + these 8, which
# are exactly OpenCV's rational distCoeffs order (k1,k2,p1,p2,k3,k4,k5,k6).
_OPENCV12_DIST_LABELS = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]


def _prompt(message: str, default: Optional[str] = None) -> str:
    """Prompt with an optional default shown in brackets."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def load_calib_config(config_path: Path, prompt_each: bool) -> dict:
    """Load calibration parameters from YAML, optionally prompting for each.

    Args:
        config_path: path to calib_config.yaml.
        prompt_each: when True, walk every field interactively using the YAML
            values as the shown defaults; when False, use the file as-is.

    Returns:
        Dict with keys image_glob, object_grid_n, object_spacing, focal,
        splined_lensmodel, fit.
    """
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    if not prompt_each:
        return cfg

    cfg["image_glob"] = _prompt("Image glob", str(cfg["image_glob"]))
    cfg["object_grid_n"] = int(
        _prompt("Chessboard inner corners per side (NxN)", str(cfg["object_grid_n"]))
    )
    cfg["object_spacing"] = float(
        _prompt("Chessboard square spacing (meters)", str(cfg["object_spacing"]))
    )
    cfg["focal"] = float(_prompt("Initial focal estimate (px)", str(cfg["focal"])))
    cfg["splined_lensmodel"] = _prompt(
        "Splined lensmodel string", str(cfg["splined_lensmodel"])
    )
    cfg["fit"] = _prompt(
        "Pinhole fit (centers-horizontal|centers-vertical|corners)", str(cfg["fit"])
    )
    return cfg


def _require_tools(*tools: str) -> bool:
    """Return True iff every named CLI tool is on PATH; print what's missing."""
    ok = True
    for tool in tools:
        if shutil.which(tool) is None:
            print(f"{tool} not found on PATH (sudo apt install python3-mrcal mrgingham).")
            ok = False
    return ok


def run_mrgingham(images_glob: str, grid_n: int, out_vnl: Path) -> bool:
    """Detect chessboard corners with mrgingham, writing a vnl corner cache.

    Args:
        images_glob: glob string mrgingham expands to find the raw images.
        grid_n: inner corners per side (symmetric NxN board).
        out_vnl: path to write the corners.vnl cache.

    Returns:
        True on success.
    """
    print(
        f"\nDetecting corners with mrgingham (gridn={grid_n}); "
        f"this can take a couple of minutes ..."
    )
    cmd = ["mrgingham", "--gridn", str(grid_n), images_glob]
    with out_vnl.open("wb") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print(f"mrgingham failed:\n{r.stderr.decode(errors='replace')}")
        return False
    # mrgingham emits one line per (image, corner); count distinct images found.
    found = len({ln.split()[0] for ln in out_vnl.read_text().splitlines()
                 if ln and not ln.startswith("#")})
    print(f"mrgingham wrote {out_vnl.name} ({found} images with detections).")
    return found > 0


def run_mrcal_calibrate(
    corners_vnl: Path,
    images_glob: str,
    lensmodel: str,
    cfg: dict,
    dest: Path,
) -> Optional[Path]:
    """Run mrcal-calibrate-cameras for one lens model and save the result.

    Args:
        corners_vnl: the shared mrgingham corner cache.
        images_glob: glob string (one camera) for imager-size discovery.
        lensmodel: mrcal lensmodel string to fit.
        cfg: parsed calib config (object_grid_n, object_spacing, focal).
        dest: destination path to copy the produced .cameramodel to.

    Returns:
        ``dest`` on success, else None.
    """
    print(f"\nSolving {lensmodel} with mrcal-calibrate-cameras ...")
    with tempfile.TemporaryDirectory(prefix="mrcal_calib_") as tmp:
        cmd = [
            "mrcal-calibrate-cameras",
            "--corners-cache", str(corners_vnl),
            "--lensmodel", lensmodel,
            "--focal", str(int(round(float(cfg["focal"])))),
            "--object-spacing", str(cfg["object_spacing"]),
            "--object-width-n", str(cfg["object_grid_n"]),
            "--object-height-n", str(cfg["object_grid_n"]),
            "--outdir", tmp,
            images_glob,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.stdout:
            print(r.stdout)
        if r.returncode != 0:
            print(f"mrcal-calibrate-cameras failed for {lensmodel}:")
            print(r.stderr)
            return None
        produced = sorted(Path(tmp).glob("camera-*.cameramodel"))
        if not produced:
            print(f"No .cameramodel produced for {lensmodel}.")
            return None
        shutil.copy(produced[0], dest)
    print(f"  wrote {dest.name}")
    return dest


def build_pinhole_artifacts(splined_path: Path, fit: str, out_dir: Path) -> None:
    """Derive the pinhole model + warp remap from the splined model.

    Writes pinhole.cameramodel, mapxy.npy, camera_matrix.pkl (pinhole K) and
    dist.pkl (zeros) into out_dir for the warp / unproject runtime modes.
    """
    model_splined = mrcal.cameramodel(str(splined_path))
    model_pinhole = mrcal.pinhole_model_for_reprojection(model_splined, fit=fit)
    mapxy = mrcal.image_transformation_map(
        model_splined, model_pinhole, intrinsics_only=True
    )
    h, w = mapxy.shape[:2]

    _, pinhole_intrinsics = model_pinhole.intrinsics()
    fx, fy, cx, cy = (float(x) for x in pinhole_intrinsics[:4])
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros((1, 5), dtype=np.float64)

    with (out_dir / "camera_matrix.pkl").open("wb") as f:
        pickle.dump(camera_matrix, f)
    with (out_dir / "dist.pkl").open("wb") as f:
        pickle.dump(dist, f)
    np.save(out_dir / "mapxy.npy", mapxy)
    model_pinhole.write(str(out_dir / "pinhole.cameramodel"))

    print(
        f"  pinhole {w}x{h} (fit={fit}): fx={fx:.3f} fy={fy:.3f} "
        f"cx={cx:.3f} cy={cy:.3f}"
    )


def _opencv12_calibrate(image_paths, grid_n, spacing, frame_size):
    """Cross-check the mrcal OPENCV12 solve with cv.calibrateCamera (rational).

    Detects a ``grid_n`` x ``grid_n`` chessboard in each image and runs
    cv.calibrateCamera with CALIB_RATIONAL_MODEL, which fits k1..k6 + p1,p2 —
    the same 8 distortion terms as mrcal's LENSMODEL_OPENCV12 — so the two
    independent solves can be compared directly.

    Args:
        image_paths: raw image paths to detect boards in.
        grid_n: inner corners per side.
        spacing: chessboard square size (meters); only affects extrinsics.
        frame_size: (width, height) of the images.

    Returns:
        (K 3x3, dist length-8 ndarray) or None if too few boards were detected.
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((grid_n * grid_n, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_n, 0:grid_n].T.reshape(-1, 2) * spacing

    objpoints, imgpoints = [], []
    for p in tqdm(image_paths, desc="opencv detect", unit="img"):
        img = cv.imread(str(p))
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, (grid_n, grid_n), None)
        if not found:
            continue
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(objpoints) < 4:
        print(
            f"OpenCV detected only {len(objpoints)} usable boards "
            f"(need >= 4); skipping cv.calibrateCamera cross-check."
        )
        return None

    rms, k, dist, _rvecs, _tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None, flags=cv.CALIB_RATIONAL_MODEL
    )
    print(
        f"cv.calibrateCamera (rational) RMS reprojection error: {rms:.4f} px "
        f"({len(objpoints)} boards used)"
    )
    return k, np.asarray(dist, dtype=np.float64).reshape(-1)[:8]


def print_opencv12_comparison(mrcal_opencv12: Path, cv_k, cv_dist) -> None:
    """Print mrcal OPENCV12 vs cv.calibrateCamera side by side."""
    _, idata = mrcal.cameramodel(str(mrcal_opencv12)).intrinsics()
    idata = np.asarray(idata, dtype=np.float64)
    cv_vec = np.array(
        [cv_k[0, 0], cv_k[1, 1], cv_k[0, 2], cv_k[1, 2], *cv_dist], dtype=np.float64
    )
    labels = ["fx", "fy", "cx", "cy", *_OPENCV12_DIST_LABELS]

    print(
        "\nOPENCV12 comparison (mrcal LENSMODEL_OPENCV12 vs "
        "cv.calibrateCamera rational):"
    )
    header = f"  {'param':<6}{'mrcal':>16}{'opencv':>16}{'Δ':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, label in enumerate(labels):
        m, c = float(idata[i]), float(cv_vec[i])
        print(f"  {label:<6}{m:>16.6f}{c:>16.6f}{c - m:>+16.6f}")


def show_graphs(model_path: Path, label: str) -> None:
    """Launch the residuals + projection-uncertainty graphs for one model.

    Each command opens an interactive gnuplot window and blocks until closed.
    Requires the model to carry optimization_inputs (mrcal-calibrate-cameras
    output does); the cv.calibrateCamera cross-check model does not, so this is
    only called on mrcal's own splined / opencv12 models.
    """
    print(f"\n=== {label}: residual directions (close the window to continue) ===")
    subprocess.run(["mrcal-show-residuals", "--directions", str(model_path)])

    print(f"=== {label}: projection uncertainty over the image ===")
    subprocess.run(["mrcal-show-projection-uncertainty", str(model_path)])

    print(f"=== {label}: projection uncertainty vs distance (center) ===")
    subprocess.run([
        "mrcal-show-projection-uncertainty",
        "--vs-distance-at", "center",
        str(model_path),
        "--set", "yrange [0:2]",
    ])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="Directory of raw chessboard images.")
    parser.add_argument("--name", help="Name of the intrinsics folder to create.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Calibration config YAML (default: calib_config.yaml beside script).",
    )
    parser.add_argument(
        "--prompt-each", action="store_true",
        help="Prompt for every calib parameter instead of using the config as-is.",
    )
    args = parser.parse_args()

    if not _require_tools("mrgingham", "mrcal-calibrate-cameras"):
        raise SystemExit(1)

    # --- inputs -----------------------------------------------------------
    image_dir_str = (
        str(args.images) if args.images else _prompt("Directory of calibration images")
    )
    image_dir = Path(image_dir_str).expanduser().resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Not a directory: {image_dir}")

    name = args.name or _prompt("Name for the new intrinsics folder")
    if not name:
        raise SystemExit("A name is required.")
    out_dir = INTRINSICS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_each = args.prompt_each
    if not prompt_each:
        ans = input(
            f"\nUse calibration config {args.config.name} as-is? "
            f"(n = prompt for each value) [Y/n] "
        ).strip().lower()
        prompt_each = ans in ("n", "no")
    cfg = load_calib_config(args.config, prompt_each)

    images_glob = str(image_dir / cfg["image_glob"])
    image_paths = sorted(image_dir.glob(cfg["image_glob"]))
    if not image_paths:
        raise SystemExit(f"No images matched {images_glob}")
    sample = cv.imread(str(image_paths[0]))
    if sample is None:
        raise SystemExit(f"Could not read {image_paths[0]}")
    h0, w0 = sample.shape[:2]
    print(f"Found {len(image_paths)} images ({w0}x{h0}); output -> {out_dir}")

    # --- detect corners once, reuse for both solves -----------------------
    corners_vnl = out_dir / "corners.vnl"
    if not run_mrgingham(images_glob, int(cfg["object_grid_n"]), corners_vnl):
        raise SystemExit("Corner detection failed.")

    # --- solve both lens models -------------------------------------------
    splined_path = run_mrcal_calibrate(
        corners_vnl, images_glob, cfg["splined_lensmodel"], cfg,
        out_dir / "splined.cameramodel",
    )
    if splined_path is None:
        raise SystemExit("Splined solve failed.")
    opencv12_path = run_mrcal_calibrate(
        corners_vnl, images_glob, "LENSMODEL_OPENCV12", cfg,
        out_dir / "opencv12.cameramodel",
    )
    if opencv12_path is None:
        raise SystemExit("OPENCV12 solve failed.")

    # --- derive pinhole artifacts (warp / unproject) ----------------------
    print("\nDeriving pinhole artifacts from the splined model ...")
    build_pinhole_artifacts(splined_path, cfg["fit"], out_dir)

    # --- opencv cross-check -----------------------------------------------
    print("\nCross-checking OPENCV12 with cv.calibrateCamera ...")
    cv_result = _opencv12_calibrate(
        image_paths, int(cfg["object_grid_n"]), float(cfg["object_spacing"]), (w0, h0)
    )
    if cv_result is not None:
        print_opencv12_comparison(opencv12_path, *cv_result)

    # --- tell the operator what to put in info.yaml -----------------------
    print(
        f"\nDone. To use this calibration, in src/info.yaml set:\n"
        f'    camera_intrinsics_path: "{name}"\n'
        f"    lens_correction_mode: warp        # remap whole frame to pinhole\n"
        f"    lens_correction_mode: unproject   # detect on raw, undistort PnP points\n"
        f"    lens_correction_mode: normal      # detect on raw, OPENCV12 dist in solvePnP\n"
        f"  (choose one lens_correction_mode; this folder supports all three.)"
    )

    # --- optional diagnostic graphs ---------------------------------------
    have_graph_tools = _require_tools(
        "mrcal-show-residuals", "mrcal-show-projection-uncertainty"
    )
    if have_graph_tools:
        ans = input(
            "\nShow residuals + projection-uncertainty graphs for the splined "
            "and OPENCV12 models? [y/N] "
        ).strip().lower()
        if ans in ("y", "yes"):
            show_graphs(splined_path, "splined")
            show_graphs(opencv12_path, "OPENCV12")

    if shutil.which("mrcal-show-projection-diff"):
        ans = input(
            "\nShow projection-diff comparison (splined vs OPENCV12)? [y/N] "
        ).strip().lower()
        if ans in ("y", "yes"):
            print("\n=== splined vs OPENCV12 projection diff ===")
            subprocess.run([
                "mrcal-show-projection-diff",
                str(splined_path), str(opencv12_path),
            ])


if __name__ == "__main__":
    main()
