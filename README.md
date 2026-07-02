# How to setup code
### ssh keys
TODO
- Download and install [git lfs](https://git-lfs.com/). This is used to manage large files for git.
- run `git clone git@github.com:TAMU-Robomasters/AutoX.git`
- run `git lfs install`
- Install [uv](https://docs.astral.sh/uv/). This is used to manage python packages. Modern version of pip or conda. 
    - Mac/Linux/WSL
    `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - Windows
    `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
> [!WARNING]
> You may have to restart your Windows machine after installing uv
- run `uv sync`

## How do I get this code to run?
- Run `. .venv/bin/activate` and `python main.py @CAMERA=WEBCAM`
- Or run `uv run main.py @CAMERA=WEBCAM`

CLI profiles use the `@OPTION=VALUE` syntax (the `@` is required). The `MCU=` profile
chooses how auto-aim reaches the devboard:

| Profile | Who owns `/dev/ttyTHS1` | Use it for |
|---|---|---|
| `@MCU=MOCK` (default) | nobody | laptop dev — identity transform, no serial, no ROS |
| `@MCU=SERIAL` | AutoX itself | running auto-aim **without** nav2 |
| `@MCU=ROS @ROS=NAV2` | the nav2 `mcu_bridge_node` | running auto-aim **with** nav2 (the sentry) |

Robot defaults (camera, backend) are persisted in `src/local_data.ignore.yaml` under
`(selected_profiles)`, so a bare `uv run main.py` picks them up.

## Running AutoX **and** the nav2 stack together

Only one process can own the MCU UART, so when both auto-aim and navigation run at once
(the sentry), the **nav2 stack owns `/dev/ttyTHS1`** and AutoX talks to the MCU *through*
nav2 over ROS (`MCU=ROS`). The bridge lives in the nav2 repo (`~/repos/nav2-sim-testing`,
`uart_odom_node` with `enable_mcu_bridge:=true`); see its `CLAUDE.md` → "MCU bridge for
AutoX". Both sides must run **460800 baud** (matches the MCU firmware).

```sh
# Terminal 1 — nav2 stack as the serial owner + MCU bridge
source /opt/ros/humble/setup.zsh
cd ~/repos/nav2-sim-testing && source install/setup.zsh
ros2 launch sam_bot_bringup real_lidar_navigate.launch.py enable_mcu_bridge:=true
#   (or just the bridge node alone, no navigation:
#    ros2 run uart_odom uart_odom_node --ros-args -p enable_mcu_bridge:=true -p baudrate:=460800)

# Terminal 2 — AutoX auto-aim, routing MCU traffic over ROS
source /opt/ros/humble/setup.zsh                 # so RosBackend can import rclpy + mcu_msgs
cd ~/repos/nav2-sim-testing && source install/setup.zsh
cd ~/repos/AutoX && uv run main.py @MCU=ROS @ROS=NAV2
```

Notes:
- AutoX **must** be started from a shell with ROS + the nav2 overlay sourced, otherwise
  the `McuDriver` child can't import `rclpy`/`mcu_msgs`.
- `@MCU=ROS` requires `@ROS=NAV2` (a startup guard enforces this).
- To run auto-aim standalone instead, stop nav2 and use `@MCU=SERIAL`.
- Quick check that only nav2 holds the port: `lsof /dev/ttyTHS1`.

# What is this repo?

This handles all main logic that runs on the Jetson.

## How does the code work?

See `CLAUDE.md` for the full architecture (engines / modules / drivers). The short
operator's view of the auto-aim engine (`src/engines/full_state_autoaim.py`):

**It learns each enemy robot's geometry once, then reuses it.** The first time it locks
onto a robot it runs in `PARAMETER_ESTIMATION`: it aims at the closest single armor panel
while measuring the robot's two panel-orbit radii and the height difference between the
panel pairs. Once those converge it saves them and switches to full-state tracking
(predicting the spinning robot's whole pose and leading the shot). A robot it has already
learned boots straight into tracking on the next run.

**Learned constants live in a JSON file** (path: `(path_to) robot_constants` in
`src/info.yaml`, default `local_data.ignore/robot_constants.json` — gitignored, persists
across restarts), keyed by robot name. To **force it to re-learn from scratch** (e.g. a new
opponent between matches), set `estimation.force_reestimation: true` in `info.yaml` (or just
delete the file). Other knobs under `estimation:` control how quickly it decides the
constants have converged — most important is `min_constant_updates` (the minimum number of
two-panel measurements it must see before trusting/saving the constants; a backstop so one
lucky frame can't trigger a premature save).

**Logging.** Code uses stdlib `logging`, routed through one queue so all the engine/driver
processes share a single ordered stream. Control it in the `log:` block of `info.yaml`:

```yaml
log:
    level: INFO          # DEBUG for verbose, WARNING to quiet down
    levels:              # per-module overrides, e.g. debug one module only:
        panel_tracking: DEBUG
    to_file: false       # also write a rotating file at (path_to) log_file
    disable_all_logging: false   # kill switch for max performance
```

The auto-aim engine logs every state transition at INFO, so watching the console tells you
when it's learning vs tracking and when it saves a robot's constants.

# Autoboot (run AutoX on every boot)

On the robot AutoX starts itself at boot via a **systemd service**, so nothing has to be
launched by hand at competition. Profiles come from `src/local_data.ignore.yaml`
(`selected_profiles`), so the boot command takes no CLI args.

```sh
./utils/setup_boot_script.sh     # install + enable autox_boot.service (restarts on crash)
./utils/kill_onboot_cv           # stop + disable it permanently
```

- **Status / logs:** `systemctl status autox_boot.service`, `tail -f ~/boot.log` (the
  wrapper `utils/autox_boot.sh` mirrors stdout there; the previous boot is kept as
  `~/boot.old.log`), or `journalctl -u autox_boot.service -f`.
- **Stop just for this session** (returns on reboot): `sudo systemctl stop autox_boot.service`.
- The service **owns the camera** and its iceoryx2 publisher. Running a second
  `uv run main.py` by hand while it's active collides on the shared-memory publisher
  (`ExceedsMaxSupportedPublishers`) and shows no frames — so `main.py` guards against this
  (`src/toolbox/autoboot_check.py`): a **manual** run aborts with instructions when the
  service is active, while the boot-launched instance is exempt (it spots the unit in its
  own `/proc/self/cgroup`). To develop with the display, stop the service first.

# At the competition
TODO

# How to Setup New Xavier 
TODO



```sh
cd repos
git clone git@github.com:TAMU-Robomasters/cv_lite.git
cd cv_lite
sudo ./run/xavier_reset_zerotier
```

Then install the autoboot service so AutoX runs on every boot (see **Autoboot** above):

```sh
./utils/setup_boot_script.sh
```
