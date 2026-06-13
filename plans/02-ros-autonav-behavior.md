# 02 — ROS bridge, AutoNav engine, behavior brain

## Goal
Let AutoX coordinate with the nav2 stack (`../nav2-sim-testing`): AutoX decides
where to navigate and whether to engage; nav2 drives to waypoints, avoids
obstacles, and reports status back. The decision logic is a `py_trees` behavior
tree living **inside** the AutoNav engine. Keep AutoX usable on a laptop with no
ROS (everything ROS is profile-gated, rclpy imported lazily).

## Why / decisions (locked)
- **Central brain, folded into the AutoNav engine**, kept as a pure rclpy-free
  `py_trees` object so it's laptop-testable and extractable later.
- AutoAim and AutoNav are otherwise dumb executors. The brain ingests devboard
  state (health/match via the MCU client), nav2 status, and the autoaim target;
  outputs waypoints (-> nav2) and an engage directive (-> AutoAim).
- **Engage is fail-OPEN**: if the brain or nav2 dies, AutoAim keeps shooting. The
  brain's engage signal is an **expiring suppression/directive**, not a required
  permission (stale -> ignored -> autonomous engage).
- **Hard local interlock = match-active only**: AutoAim refuses to aim/fire when
  the match is not active (pre-match OR post-match-end), read locally from the
  devboard via its own `McuClient`. Robot-dead/respawn and referee-disable are NOT
  local interlocks (brain policy / referee hardware).
- **Waypoints via the `nav2_msgs/NavigateToPose` action** (goal+feedback+result+
  cancel), not `/goal_pose`. `square_patrol.py` in the nav2 repo is the reference.
- **"Wiggle at waypoint" via small relative NavigateToPose goals**, NEVER `/cmd_vel`
  (nav2 owns `/cmd_vel`; `uart_odom_node` forwards it to the MCU with a 0.3s
  watchdog — a direct velocity would fight it). Robot is holonomic,
  `yaw_goal_tolerance > π`, so wiggle = translational offsets.

## Design (AutoX side)
- `src/drivers/ros_bridge.py` `RosBridge(Driver)`, profile-gated (`ROS=NAV2`).
  **Lazy-imports `rclpy`/`nav2_msgs`/`geometry_msgs` in `initialize()`** (child).
  Spins an rclpy node; bridges nav2 <-> AutoX. ROS<->AutoX payloads are small
  fixed-size (no shared memory). Isolate the embedded<->ROS axis mapping
  (`odom_x=emb_y`, `odom_y=-emb_x`; see nav2 `uart_odom_node.publish_transformed_odom`).
- `src/engines/autonav.py` `AutoNav(Engine)`:
  - declares `drivers = {"mcu": McuDriver, "ros": RosBridge}` (and `frames` only if
    it needs vision; usually not).
  - holds the `py_trees` tree (plain `py_trees`, NOT `py_trees_ros`).
  - reads nav2 state + devboard via handles; writes waypoints to `RosBridge`,
    writes the engage directive to AutoAim (cross-engine — see plan 03, or a
    dedicated small queue/iceoryx field in the interim).
- AutoAim changes: read the engage directive (with a freshness timestamp -> fail
  open) and enforce the match-active interlock locally via its `McuClient`.

## Design (nav2 side, `../nav2-sim-testing`)
- Extend `src/uart_odom/uart_odom/uart_odom_node.py` into an `mcu_bridge_node`
  (same process, same serial port — only one node may open `/dev/ttyTHS1`), gated
  by a launch param so odom-only mode is unchanged. Add, for `MCU=ROS`: a
  transform-query service + a firing-solution sink (port `QueryToEmbedded`/
  `EmbeddedTransformationMessage`/`send_angles_to_embedded`).
- Expose `NavigateToPose` (already there via `bt_navigator`/`nav2_simple_commander`)
  to the bridge + a compact status topic (mode/pose/idle/moving/arrived).
- DO NOT touch the holonomic/translation-only DWB config, `OmniMotionModel`,
  `yaw_goal_tolerance > π`, or `spin` (the nav2 CLAUDE.md warns against it).

## nav2 watchdog / revive (future, design only for now)
A supervisor that launches the nav2 stack, pings it for liveness (node/topic
presence or a heartbeat), and runs a restart sequence on failure. Because the
brain is folded into AutoNav (same crash domain as nav2), rely on the fail-OPEN
engage default so AutoAim keeps working while nav2 restarts.

## Profile
`ROS=NONE` (default) / `ROS=NAV2`. `ROS=NAV2` implies a sourced ROS env; match
`RMW_IMPLEMENTATION` + `ROS_DOMAIN_ID` between AutoX and nav2. The AutoX venv
already has `include-system-site-packages=true`, so `import rclpy` works when ROS
is sourced — but never add rclpy/ros to `pyproject.toml`.

## Verification
- Laptop `ROS=NONE`: AutoNav ticks the py_trees tree against mocked nav state; no
  `rclpy` imported (assert `rclpy not in sys.modules`).
- Jetson `ROS=NAV2` (after `source /opt/ros/humble/setup.bash` + the nav2 stack):
  AutoX sends a waypoint, robot navigates, status returns, "arrived" triggers a
  wiggle; with `MCU=ROS`, only the nav2 node opens `/dev/ttyTHS1`.

## Notes
- iceoryx2 does NOT bridge to ROS in practice (`rmw_iceoryx2` is v0.1/single-host),
  so the AutoX<->nav2 channel is rclpy in `RosBridge`, not shared memory.
- `py_trees` (not `py_trees_ros`) keeps the brain uv-clean and laptop-runnable.
