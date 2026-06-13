# 01 — MCU bridge (queue-RPC driver)

## Goal
One process owns the UART to the devboard; any engine talks to it through a thin
client with a clean API, so multiple engines can share the MCU. Keep the
request/response style PF already uses (`get_transformation(timestamp)`,
`send_solution(...)`). Backends selectable by profile: `SERIAL | ROS | MOCK`.

## Why
Today `ParticleFilterAutoAimEngine` constructs `EmbeddedCommunicator()` directly
and opens `/dev/ttyTHS1` in-process. Only one process can hold that serial port,
so no other engine (or the nav2 stack) can use the MCU. The devboard also holds a
**fine-resolution transform buffer** — the driver does NOT buffer; it forwards the
caller's timestamp and returns the devboard's answer.

## Design
Follow the same Driver contract `CameraDriver` uses (`provision/from_conn/client`),
but the transport is a `multiprocessing.Queue` RPC instead of iceoryx2 (control
plane, tiny messages, no external dep):

- `src/drivers/mcu.py`:
  - `McuDriver(Driver)` — owns the UART backend in its child process. `execute()`
    drains a request queue, services each request, replies on the per-request
    response queue. `provision(name)` returns `{"req": Queue(), "resp": Queue()}`
    (created in the parent so they survive fork); `client(conn)` returns an
    `McuClient(conn["req"], conn["resp"])`.
  - `McuClient` — the handle engines hold. API:
    - `get_transformation(timestamp, timeout=...) -> (yaw, pitch, 4x4) | None`
    - `send_solution(pitch, yaw, time_until_fire, cv_state)` (fire-and-forget)
    - `get_match_state() -> ...` (for AutoAim's local interlock; see plan 02)
    It pushes a request `{op, args, reply_to}` and blocks on `reply_to.get(timeout)`.
  - Backends (a small protocol + 3 impls):
    - `SerialBackend` — wraps the existing `src/subsystems/embedded_communicator.py`
      **verbatim** (reuse `get_camera_to_ballistic_transformation`,
      `send_angles_to_embedded`). No protocol change.
    - `RosBackend` — when `MCU=ROS`, forwards to nav2's serial-owning node (the
      `uart_odom_node` extended into an `mcu_bridge_node`); see plan 02.
    - `MockBackend` — canned/identity transform + replay; laptop dev.
- `src/types/rpc.py` — small `@dataclass` `McuRequest`/`McuResponse` envelopes.

## Profile (`src/info.yaml` + `src/toolbox/globals.py`)
Add an `mcu` block to `(default)` (`backend`, `rpc_timeout`) and profiles
`MCU=SERIAL|ROS|MOCK`. Default `MCU=MOCK` so laptops run with no hardware. Factory
must enforce `MCU=ROS ⇒ ROS=NAV2`.

## Steps
1. `McuDriver`/`McuClient`/backends + `rpc.py`. `SerialBackend` just delegates to
   `EmbeddedCommunicator`.
2. Add `mcu` profiles + globals defaults.
3. Migrate `ParticleFilterAutoAimEngine`:
   - declare `drivers = {"frames": CameraDriver, "mcu": McuDriver}`.
   - in `initialize()`: `self.mcu = self.driver("mcu")` (drop the inline
     `EmbeddedCommunicator()`).
   - replace `self.communicator.get_camera_to_ballistic_transformation(frame_delay_ms)`
     with `self.mcu.get_transformation(self.ctx.frame_ts)` (pass the timestamp;
     the devboard does the buffered lookup), and `send_angles_to_embedded(...)` in
     `update()` with `self.mcu.send_solution(...)`.
4. `launch_system` already provisions/wires it (it spawns one shared `McuDriver`).

## Verification
- Laptop `MCU=MOCK`: PF runs end-to-end against canned transforms; no serial opened.
- Jetson `MCU=SERIAL`: bench against the devboard; confirm transform query + solution
  send reproduce current behavior; verify a second engine (e.g. detection test) can
  also hold an `McuClient` to the same driver without serial contention.
- Unit test the RPC round-trip + timeout with `MockBackend`.

## Gotchas
- Queues must be created in the parent (`provision`) and passed to both the driver
  and the engine; they survive fork. Open the serial port in the child
  (`McuDriver.initialize()`), never `__init__`.
- The devboard owns the time buffer — don't add a ring buffer in `McuDriver`.
- `EmbeddedCommunicator._setup_serial_port` shells out to `sudo chmod` using
  `~/.pass`; keep that behavior in `SerialBackend`.
