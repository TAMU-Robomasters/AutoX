#!/usr/bin/env bash
# nav2 service entrypoint: source ROS, colcon-build the mounted workspace, exec.
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"

# The nav2-sim-testing workspace is mounted at /ws (read-write). Build it once
# per fresh container; --symlink-install means launch/config/yaml edits are then
# live without a rebuild (see nav2-sim-testing CLAUDE.md).
if [ -d /ws/src ]; then
  # The public laserscan-merger submodule is needed by the sim launches; the
  # private SSH dual-ldlidar submodule is hardware-only and intentionally skipped.
  if [ ! -e /ws/src/multi-laserscan-toolbox-ros2/package.xml ]; then
    echo "[nav2-entrypoint] multi-laserscan-toolbox-ros2 submodule missing; trying to init…"
    git -C /ws submodule update --init src/multi-laserscan-toolbox-ros2 \
      || echo "[nav2-entrypoint] WARN: submodule init failed (offline/private?); building what's available"
  fi

  if [ ! -f /ws/install/setup.bash ]; then
    echo "[nav2-entrypoint] colcon build (ignoring hardware-only ldlidar packages)…"
    ( cd /ws && colcon build --symlink-install --packages-ignore-regex '.*ldlidar.*' ) \
      || echo "[nav2-entrypoint] WARN: colcon build reported errors; continuing with what built"
  fi
  [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
fi

exec "$@"
