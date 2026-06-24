# 09 — Sentry Nav2 behavior + AutoX-owned UART

Supersedes the **UART direction** of `02-ros-autonav-behavior.md` (the AutoNav
brain + engage design in 02 still holds; only *who owns the serial* flips). Read
02 for the brain rationale, this for the transport + the laptop test rig.

## Goal
Run the sentry's nav2 behavior — a `py_trees` brain that sends waypoints to nav2
and an engage directive to auto-aim — **without adding latency to the auto-aim
path**, and make the whole loop runnable on a laptop (no Jetson/lidar/devboard)
via mocks + a Docker ROS 2 Humble + Nav2 + Gazebo rig.

## Why invert UART ownership
One physical UART (`/dev/ttyTHS1`, 460800) multiplexes four message types by a
`messageType` byte (see `embedded_communicator.py` + nav2 `uart_odom_node.py`):

| msg | payload | consumer | latency |
|---|---|---|---|
| `'t'` transform | 3B → 69B | **auto-aim, per frame** | **extreme** |
| `'d'` solution | 12B f&f | **auto-aim, per frame** | **extreme** |
| `'q'` odom | 2B → 13B | nav (100 Hz) | tolerant |
| `'v'` velocity | 10B f&f | nav (`/cmd_vel`) | tolerant |

Plan 02 has **nav2 own the port**; AutoX reaches `'t'`/`'d'` over a ROS service
(`mcu.py:RosBackend` → `/mcu/query_transform`) — a DDS round-trip on every aim
frame, queued behind nav2's 100 Hz odom poll on a single-threaded executor.

**Invert it:** AutoX's `McuDriver` owns the UART; `'t'`/`'d'` stay on the existing
in-process queue-RPC (no ROS). nav2 reaches the MCU through a ROS bridge **AutoX
hosts** (publishes `/odom`+TF, consumes `/cmd_vel` → `'q'`/`'v'`). Only
latency-tolerant traffic crosses ROS. Worst-case aim-path serial latency = one
in-flight message (~1.5 ms), bounded and low-jitter. The MCU control plane stays
on the queue-RPC `McuDriver` (plan 01 decision; iceoryx2 is frames-only).

**Canonical real-hw profile becomes `MCU=SERIAL ROS=NAV2 @SENTRY`.** `MCU=ROS`
(nav2-owns) is **deprecated**, kept as a fallback.

## Design (AutoX)
1. **`mcu.py` full ownership.** Add `get_odometry`(`'q'`)/`send_velocity`(`'v'`) to
   the backend protocol + `SerialBackend`/`MockBackend` (and a no-op deprecated
   `RosBackend`); port the `'q'`/`'v'` structs into `embedded_communicator.py`
   (mirror `uart_odom_node.py`). `McuDriver` gets a **priority lane** (two request
   queues: hi = transform/solution, lo = odom/velocity). Fix the **multi-consumer
   response queue** (the inversion adds a 2nd concurrent consumer; the `McuClient`
   docstring already flags this): one response queue per consumer, allocated in
   `provision()` (plain fork-inherited `mp.Queue` to keep the fast path fast),
   selected by a `consumer_key` in the request envelope (`types/rpc.py`).
   `MockBackend` integrates `send_velocity` → `get_odometry` (kinematic odom, so
   `/cmd_vel` moves the mock with no Gazebo) + returns match-active for
   `get_match_state`.
2. **`engines/mcu_bridge.py` `McuBridgeEngine`** — the inverted `uart_odom_node`.
   `drivers={"mcu": McuDriver}`, lazy rclpy, gated on `ros.stack=="nav2"`. Loop:
   `spin_once` (service `/cmd_vel`) → odom-interval `mcu.get_odometry()` → publish
   `/odom`, `/embedded_odom`, static `embedded_odom→odom` + `odom→base_link` TF
   (port verbatim). `/cmd_vel` → rotate by θ → `mcu.send_velocity` (+0.3 s
   watchdog).
3. **`engines/autonav.py` `AutoNav`** — pure `py_trees` brain (rclpy-free) +
   `NavigateToPose` action client + nav status sub + devboard reads via `McuClient`;
   outputs waypoints (action; wiggle = small relative goals, never `/cmd_vel`) and
   the engage directive (cross-engine queue). Add `py_trees` to `pyproject.toml`.
4. **Multi-link cross-engine queues** (`core/engine.py` + `orchestrator.py`).
   Generalize the single `publishes_queue`/`subscribes_queue` to named multi-links
   so AutoAim can subscribe to **both** `circlet_detections` and `engage_directive`;
   keep the one-publisher-per-name hard error; `launch_system` launches
   McuBridge+AutoNav when `ros.stack=="nav2"`.
5. **AutoAim** (`full_state_autoaim.py`): subscribe `engage_directive` (**fail-OPEN**
   + freshness), and a **match-active** hard interlock via `mcu.get_match_state()`
   in `_publish_solution`.
6. **Config** (`info.yaml` + `globals.py`): `ROS=NAV2` now launches AutoX's bridge
   (not nav2-owns-serial); document `MCU=SERIAL ROS=NAV2`; deprecate `MCU=ROS`; add
   an `autonav:` block (waypoints, engage staleness, wiggle, odom rate,
   `cmd_vel_timeout`).

## Design (nav2-sim-testing — minimal)
Run `real_lidar_navigate.launch.py use_uart_odom:=false`; **AutoX's McuBridge is
the `/odom` + `odom→base_link` TF source**. Mark `enable_mcu_bridge` +
`mcu_msgs/{QueryTransform,FiringSolution}` deprecated (legacy `MCU=ROS`). Gazebo
`bringup.launch.py` reused as-is for rig 1.

## Docker test rig (`docker/`)
`ros:humble` base (Ubuntu 22.04 → Py 3.10, matches AutoX + Jetson). Two services
(`nav2`, `autox`) on `network_mode: host` + shared `ROS_DOMAIN_ID`/`RMW`; sources
mounted + built at start (`colcon` / `uv sync --no-install-project --system-site
rclpy`). See `docker/README.md`. **Two rigs:** (1) Gazebo full sim — Gazebo owns
odom/scan/cmd_vel, AutoX runs AutoNav only (brain/nav plumbing); (2) MCU-inversion
harness — `McuBridgeEngine` ⇄ `MockBackend` kinematic odom + a latency harness
proving bounded aim-path latency under 100 Hz odom load. ⚠ Disk: a full build is
several GB; check `df -h /` first.

## Build order
0. Docker rig (done). 1. `mcu.py` ops + priority + per-consumer queues + mock.
2. `McuBridgeEngine`. 3. multi-link queues. 4. `AutoNav` + brain. 5. AutoAim
engage/interlock. 6. nav2 launch/docs. 7. rigs 1+2 end-to-end.

## Verification
- **Unit (host, no ROS/mrcal):** brain ticks vs mocked nav state + `assert "rclpy"
  not in sys.modules`; `McuDriver` priority + multi-consumer round-trip/timeout;
  `MockBackend` kinematic odom; multi-link wiring (duplicate-publisher raises).
- **Rig 1 (Gazebo):** AutoNav sends NavigateToPose, robot drives, status returns,
  engage flips AutoAim `cv_state`.
- **Rig 2 (MCU):** McuBridge publishes `/odom` from mock; teleop `/cmd_vel` moves
  it; latency harness shows bounded aim-path query time under odom load.
- **Jetson (future):** `MCU=SERIAL ROS=NAV2 @SENTRY` — AutoX owns serial, nav2
  navigates on AutoX odom, auto-aim latency unaffected by nav.

## Gotchas
- `py_trees` (not `py_trees_ros`) — keep the brain uv-clean + laptop-runnable.
- `get_match_state` has no wire message yet — mocked; flagged for firmware.
- AutoX container uses `uv sync --no-install-project` to skip the CUDA `pf_cuda_cv`
  build (unneeded for nav engines) and `--inexact` to keep system-site `rclpy`.
- Engines that declare `drivers`/queues must launch via `launch_system` (per plan
  07). Never open serial/rclpy in `__init__` (child-only `initialize()`).
