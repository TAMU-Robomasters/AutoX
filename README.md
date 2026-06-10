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
TODO

# At the competition
TODO

# How to Setup New Xavier 
TODO



```sh
cd repos
git clone git@github.com:TAMU-Robomasters/cv_lite.git
cd cv_lite
sudo ./run/xavier_reset_zerotier
sudo ./run/xavier_setup_boot_script.js
```
