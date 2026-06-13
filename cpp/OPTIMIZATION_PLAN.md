# Detection Pipeline C++ Port — Plan

## Build commands

Run from `cpp/`:

```bash
# Configure (only needed when CMakeLists.txt or presets change)
cmake --preset cv -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)

# Build
cmake --build out/build/cv -j4

# Install the .so to the project root so `import armor_panel_cpp` works
cmake --install out/build/cv
```

## What you're doing

Two efforts toward the same goal — lower per-frame latency in the detection pipeline:

1. **Rewrite the body of `main()` in C++.** The whole detection chain ends up behind one pybind11 call returning `List[Panel]`. Outer loop, PF, ballistics, and `embedded_communicator` stay in Python.
2. **Accelerate GPU-friendly ops with VPI.** Threshold, morph, warp, resize, grayscale, and (eventually) NVMM ingestion go on CUDA / PVA / VIC backends. VPI lives inside the C++ module because that's where the pipeline lives — but the language is incidental; if a VPI op is easier to prototype in Python first, do that.

Both are required to hit the latency target. The order in this plan does the C++ port first, then layers VPI on top — because a working CPU baseline gives you something to benchmark VPI against, and because VPI calls are simpler to wire up once the surrounding code is already C++.

## The boundary

| Side | What lives here |
|------|----------------|
| C++ | `get_frame` → `frame_process` → `bounding_boxes` → `pairing` → `armour_corners` → `icon_detection` → `get_cord` |
| Python | Outer loop, particle filter, `theta_solver`, `embedded_communicator`, ballistics, debug drawing |
| Crosses the boundary | One `detect_panels()` call per frame. Returns `List[Panel]`. |

Frames never leave C++. `Lights`, contours, and `cv::Mat` are C++-internal — no Python bindings for them. Only `Panel` is exposed to Python.

## Why this shape

- One pybind11 call per frame costs microseconds. Negligible.
- VPI's C++ API is the native one. No need to wrap NVMM/DMABUF/EGLImage through Python.
- PF, ballistics, and comms are not on the hot path. Rewriting them is dev cost with zero perf win.

---

## File structure

A top-level `cpp/` directory mirroring `subsystems/`:

```
cpp/
├── CMakeLists.txt
├── bindings.cpp                 # pybind11 module definition
├── detector.h / detector.cpp    # top-level detect_panels() pipeline
└── subsystems/
    ├── types.h                  # Lights, Panel structs (header-only)
    ├── video_source.h / .cpp    # get_frame, init
    ├── frame_process.h / .cpp   # threshold + morph + findContours
    ├── armor.h / .cpp           # bounding_boxes, pairing, armour_corners
    ├── icon_detection.h / .cpp  # warp + resize + classify
    └── pnp.h / .cpp             # get_cord (solvePnP)
```

**Why this layout, not the conventional `include/` + `src/` split:**
- One-to-one with Python `subsystems/`, so the mapping stays obvious.
- This is an internal single-binary module, not a public library — no need to expose headers separately.
- "Where does X live?" is always the same filename in either tree.

`Panel` is the only struct that crosses to Python. It gets its `py::class_` binding in `bindings.cpp`.

---

## Stage 0 — Toolchain skeleton

**Goal:** Python can `import armor_panel_cpp` and call `detect_panels()` returning an empty list.

**Steps:**
1. Create `cpp/CMakeLists.txt`. Find `pybind11`, `OpenCV`, `VPI` (locate but don't link VPI yet).
2. Add empty `detector.cpp` with `std::vector<Panel> detect_panels() { return {}; }` and a stub `Panel`.
3. Write `bindings.cpp` exposing `detect_panels` and a minimal `Panel` (one field is enough).
4. Build, install into the project. Import it from `main.py`.

**Done when:** `python -c "import armor_panel_cpp; print(armor_panel_cpp.detect_panels())"` prints `[]`.

**Footguns:**
- pybind11 and OpenCV must be built against the same Python you're running. Use `python3 -m pybind11 --cmakedir`.
- On Jetson the system OpenCV may not have every module. Confirm `cv::findContours` compiles and links before moving on.

---

## Stage 1 — Port `pairing`

**Goal:** Python's `pairing()` is replaced by a C++ implementation. Same input, same output.

**Source:** [subsystems/armor.py:63-137](subsystems/armor.py#L63-L137)

**Steps:**
1. Define `Lights` in `cpp/subsystems/types.h` matching the Python class fields.
2. Implement `pair_lights(const std::vector<Lights>&) -> std::vector<std::pair<Lights, Lights>>` in `armor.cpp`.
3. Single O(N²) scan, no heap allocations. Mirror the greedy argmin selection from Python.
4. Bind `Lights` and `pair_lights` in `bindings.cpp`.
5. In `main()`, swap `pairing()` for the C++ version. Leave everything else unchanged.

**Done when:** Output matches Python `pairing` on the same inputs. Loop time drops.

**Footguns:**
- Python's `set` for "indices already used" is just a `std::vector<bool>` in C++. Don't overbuild.
- N is ~10. Anything more clever than O(N²) is wasted code.

---

## Stage 2 — Absorb `bounding_boxes` + `armour_corners`

**Goal:** A single C++ call takes the frame + contours and returns `List[Panel]`. `frame_process` is still called from Python (we'll move it next).

**Sources:** [subsystems/armor.py:29-61](subsystems/armor.py#L29-L61), [subsystems/armor.py:139-177](subsystems/armor.py#L139-L177)

**Steps:**
1. Add `Panel` to `types.h` with its full fields (`tvec`, `rvec`, corners, `id`, `yaw`, etc.).
2. Write a `numpy_to_mat()` helper — zero-copy `cv::Mat` wrap of a `py::array_t<uint8_t>`.
3. Implement `contours_to_panels(contours, frame)` in `detector.cpp`: bounding_boxes → pair_lights → armour_corners.
4. Bind `Panel` so Python can read `.tvec`, `.rvec`, `.yaw`, etc.
5. Rewire `main()` to call this for the whole chunk.

**Done when:** Panels at this point in the pipeline match Python output on the same frame.

**Footguns:**
- Frame from the current `video_source.py` is `BGRx` (4 channels) → `CV_8UC4`, not `CV_8UC3`. Confirm with `frame.shape[2]`.
- Don't write into the numpy-wrapped `Mat` past the function's return — that buffer belongs to Python.
- OpenCV Python contours are `(N, 1, 2)` int32 arrays. Convert each to `std::vector<cv::Point>` rather than fighting numpy buffer shapes.

---

## Stage 3 — Move `frame_process` into C++

**Goal:** Contours never cross the boundary. `detect_panels(frame, enemy_color)` runs threshold + morph + `findContours` internally.

**Source:** [subsystems/frame_proccesing.py:7-27](subsystems/frame_proccesing.py#L7-L27)

**Steps:**
1. Add `frame_process.cpp` with `find_contours(const cv::Mat& frame, EnemyColor) -> std::vector<std::vector<cv::Point>>`.
2. Use plain OpenCV C++ — no VPI yet. CPU baseline first.
3. Wire it into `detect_panels` so it runs before the Stage 2 pipeline.
4. Remove `fraame_process` from `main()`.

**Done when:** `main()` is down to roughly `frame = get_frame(); panels = detect_panels(frame, enemy_color)`.

---

## Stage 4 — Move `icon_detection` into C++

**Goal:** Icon classification runs inside the C++ pipeline. `icon_stack` is loaded once at module init.

**Source:** [subsystems/icon_detection.py:20-44](subsystems/icon_detection.py#L20-L44)

**Steps:**
1. At module init, load `icon_stack` from the same files Python reads. Store as `std::vector<cv::Mat>`.
2. Port the warp → grayscale → resize → adaptive threshold → XOR chain.
3. Hook into `detect_panels` between `armour_corners` and `get_cord`.

**Footguns:**
- `cv::adaptiveThreshold` is native in C++ with the same signature. No need to emulate.
- Confirm `icon_stack` load order matches Python so panel IDs line up.

---

## Stage 5 — Move `get_cord` (PnP) into C++

**Goal:** End-to-end detection inside C++ — the only thing crossing in is still `frame`.

**Source:** [subsystems/PnP.py:22-55](subsystems/PnP.py#L22-L55)

**Steps:**
1. Load `cam_matrix` and `dist` once at module init.
2. Port `get_cord` — `cv::solvePnP` per panel. Consider batching only if it pays.
3. Delete the dead `if panel.id != 0` branch — both arms are identical.
4. Store `tvec` / `rvec` as `cv::Vec3d` on the `Panel` struct.

**Done when:** Python `main()` is `frame = get_frame(); return detect_panels(frame, enemy_color)`.

---

## Stage 6 — Move `get_frame` into C++

**Goal:** Frame never crosses the boundary. `detect_panels(enemy_color)` takes no arguments aside from color.

**Source:** [subsystems/video_source.py](subsystems/video_source.py) — only the `BufferlesCvCapture` / `USB_CAM` path is live. Ignore `REALSENSE` and the fallback branches.

**Steps:**
1. Port the `VideoCapture`-based path that currently works without sudo.
2. Expose `video_source_init()` from C++ so init stays explicit.
3. `detect_panels(enemy_color)` calls `get_frame` internally.
4. Decide now whether you want a `get_last_frame() -> numpy` accessor for debug visualization, since `cv.imshow` in `main.py` won't otherwise have a frame.

**Done when:** Python `main()` body is one line: `return detect_panels(enemy_color)`.

---

## Stage 7 — VPI for `frame_process`

**Goal:** Threshold and morph close run on GPU (CUDA or PVA backend). Frame stays GPU-resident until `findContours`.

**Source:** the C++ `frame_process.cpp` you wrote in Stage 3.

**Steps:**
1. Add VPI linking to `CMakeLists.txt`.
2. On the detector struct, hold a reusable `VPIStream`, plus `VPIImage` handles for the input frame, threshold output, and morph output. **Allocate once at init, not per frame.**
3. Replace `cv::threshold` with `vpiSubmitThreshold`.
4. Replace `cv::morphologyEx(MORPH_CLOSE)` with `vpiSubmitDilate` → `vpiSubmitErode`.
5. `vpiStreamSync` once, then wrap the result as `cv::Mat` for `cv::findContours`.

**Done when:** The contours produced are visually equivalent to the CPU version and loop time improves.

**Footguns:**
- `findContours` has no VPI equivalent. The GPU→CPU copy lives here. Measure its cost; don't pursue custom CUDA connected-components unless that copy is the actual bottleneck.
- Try both CUDA and PVA backends — PVA is often faster for these ops on Orin because it frees CUDA for other work.

---

## Stage 8 — VPI for `icon_detection`

**Goal:** Perspective warp + grayscale + resize run on the VIC backend (essentially free).

**Source:** the C++ `icon_detection.cpp` you wrote in Stage 4.

**Steps:**
1. Reuse the `VPIStream` from Stage 7. Add `VPIImage` handles for warped / grayscale / resized outputs.
2. Replace `cv::warpPerspective` with `vpiSubmitPerspectiveWarp` (VIC backend).
3. Replace `cv::cvtColor(BGR2GRAY)` with `vpiSubmitConvertImageFormat` (VIC backend).
4. Replace `cv::resize` with `vpiSubmitRescale` (VIC backend).
5. Leave `cv::adaptiveThreshold` and the `bitwise_xor` against `icon_stack` on CPU — VPI has no direct adaptive threshold, and the XOR is tiny.

**Done when:** Classification output matches CPU version; icon stage time drops.

**Footguns:**
- The VIC backend has strict format / alignment requirements. Convert to a VPI-friendly format up front rather than fighting per-op format mismatches.
- Don't try to emulate `cv::adaptiveThreshold` with VPI box filter + subtract + threshold unless benchmarks say it pays. Usually doesn't.

---

## Stage 9 — VPI NVMM ingestion (only if reviving GStreamer)

**Goal:** Frame is born on GPU. No CPU copy at acquisition time.

**Source:** the C++ `video_source.cpp` you wrote in Stage 6.

**Prerequisite:** The GStreamer-with-sudo issue from the `da4e0e5` commit must be solved first. That's a runtime permissions problem, not a port problem — fight it on its own branch before starting this stage.

**Steps:**
1. Build a GStreamer pipeline that terminates in `appsink` with `video/x-raw(memory:NVMM)`.
2. In C++, pull the NVMM buffer and wrap it as a `VPIImage` via DMABUF fd (NvBufSurface API).
3. Feed the `VPIImage` directly to Stage 7's threshold instead of converting through CPU.

**Done when:** No `cv::Mat` is allocated host-side at frame acquisition; loop time drops by the cost of one full-frame memcpy.

**Footguns:**
- NvBufSurface APIs are L4T-specific. Check `/opt/nvidia/jetson` headers, not generic GStreamer docs.
- This is the highest-effort stage with the smallest marginal win (one memcpy per frame). Decide based on Stage 7/8 numbers whether it's worth your time.

---

## Skip list

| Module | Why |
|--------|-----|
| `subsystems/communicate.py` | Already `ctypes.Structure` with C-level layout. |
| `subsystems/shot_timing.py` | Mostly dead code. |
| `subsystems/draw.py` | Debug viz only — Python is fine. |
| `subsystems/pf.py` | Separate C++ branch. Not your problem here. |
| `theta_solver`, `alignment_time` in `main.py` | A handful of float ops, not a bottleneck. |

---

## How to know each stage worked

Before moving to the next stage:
1. Diff output panels against a saved Python baseline on the same frame.
2. Read `Loop time:` from [main.py:220](main.py#L220). Compare against the previous stage.
3. Run a real session and watch for regressions.

If a stage doesn't improve loop time, find out why before continuing — the plan assumes each stage compounds.
