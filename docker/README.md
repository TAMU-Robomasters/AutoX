# Sentry nav sim rig (ROS 2 Humble + Nav2 + Gazebo + AutoX)

Runs the full sentry nav loop on a laptop with **no Jetson, no lidar, no
devboard** — everything is mocked or simulated, in containers, sharing one DDS
graph. Built for plan **`plans/09-sentry-nav2-behavior.md`** (the AutoX-owned-UART
inversion + the AutoNav behavior brain).

## TL;DR

```sh
cp .env.example .env            # then edit if needed
make df                         # CHECK FREE DISK FIRST (see "Disk" below)
make build                      # build both images (several GB, one-time)

# Rig 1 — Gazebo full sim (brain / nav plumbing):
make sim                        # terminal A: Gazebo + Nav2 (odom+scan+cmd_vel)
make brain                      # terminal B: AutoX AutoNav -> NavigateToPose goals

# Handy:
make shell-nav2 / make shell-autox / make topics / make teleop / make down / make clean
```

## Why this shape

| | Owner | Python | Notes |
|---|---|---|---|
| `nav2` service | `ros:humble-ros-base` + Nav2 + Gazebo | 3.10 | colcon-builds the mounted `../../nav2-sim-testing` at start |
| `autox` service | `ros:humble-ros-base` + uv + AutoX | 3.10 | `uv sync` into a `--system-site-packages` venv at start |

- **Humble base for AutoX too.** Humble's `rclpy` is built for Python 3.10 = AutoX's
  pin = the Jetson runtime. There is no PyPI `rclpy`; a system-site-packages venv is
  the only clean way to import it. This image mirrors the Jetson, and `MCU=MOCK` +
  `@MOCK_CAM` mean no serial / camera / mrcal are needed.
- **Sources are mounted, not baked.** Both entrypoints build at container start
  (`colcon build --symlink-install` / `uv sync`), so host edits are live and the
  private SSH `dual-ldlidar` submodule never needs to be present.
- **Shared DDS over `network_mode: host` + `ipc: host`.** Both services (and any
  host-side ROS) discover each other via the same `ROS_DOMAIN_ID` /
  `RMW_IMPLEMENTATION` (`.env`).

## Disk ⚠️

A full build creates several GB (Humble base ~1.5 GB, Nav2+Gazebo ~2–3 GB, the
AutoX deps incl. an iceoryx2 source build ~2–3 GB). **Check headroom first:**

```sh
df -h /          # this laptop was ~12 GB free at authoring time — tight
docker system df # reclaimable space
```

If low: `docker builder prune -af && docker image prune -af`, or move Docker's
data-root to a bigger disk (`/etc/docker/daemon.json` → `"data-root": "/path"`,
then restart docker), or build just one service at a time (`make build-nav2`).

## The two test rigs

**Rig 1 — Gazebo full sim (brain / nav plumbing).** Gazebo owns `/odom`, `/scan`,
and the `/cmd_vel` sink; AutoX runs **AutoNav only**. Validates the `py_trees`
brain, `NavigateToPose` goals, status feedback, and the engage directive. The MCU
is *not* in this loop.

```sh
make sim     # nav2: ros2 launch sam_bot_bringup bringup.launch.py
make brain   # autox: AutoNav drives NavigateToPose goals
```

**Rig 2 — MCU-inversion harness (latency proof).** No full Nav2. Exercises
`McuBridgeEngine` ⇄ `MockBackend` kinematic odom: AutoX publishes `/odom` from the
mock, `teleop` drives `/cmd_vel`, the mock odom moves. A latency harness then
hammers `'t'` transform queries under a concurrent 100 Hz odom load and asserts the
aim-path latency stays bounded (the whole point of AutoX owning the UART). This rig
arrives with plan-09 Phase 1–2/7 (it needs `McuBridgeEngine`); the container is
ready for it now.

The two rigs only fully merge on real hardware (AutoX owns the real serial; real
lidar feeds Nav2).

## GUI / GPU

Headless by default (software GL). For Gazebo/RViz windows:

```sh
xhost +local:docker
export DISPLAY=$DISPLAY            # put in .env
# GPU GL: set LIBGL_ALWAYS_SOFTWARE=0 and add the nvidia runtime to the nav2
# service (this box has an RTX 3070): `runtime: nvidia` + NVIDIA_VISIBLE_DEVICES=all
# (requires nvidia-container-toolkit on the host).
```

## Known iteration points (first real build)

- **Submodules.** The sim launch uses `multi-laserscan-toolbox-ros2` (public
  https). The entrypoint tries `git submodule update --init` for it; the private
  SSH `dual-ldlidar` is hardware-only and skipped (`--packages-ignore-regex
  '.*ldlidar.*'`). If `bringup.launch.py` references a node from an unbuilt package,
  adjust the entrypoint's package selection.
- **`uv sync`.** Uses `--no-install-project` (skips the CUDA `pf_cuda_cv` build,
  unneeded here) and `--inexact` (keeps system-site `rclpy`). If `iceoryx2` fails to
  build, confirm `cargo`/`rustc` are present (they are, in the image).
- **Sim params.** `nav2_params.yaml` (Gazebo) differs from the real-lidar config;
  see `nav2-sim-testing/CLAUDE.md`.
