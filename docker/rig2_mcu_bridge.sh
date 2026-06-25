#!/usr/bin/env bash
# Rig 2 (plan 09): McuBridgeEngine <-> MockBackend kinematic odom.
# Launch the bridge, confirm /odom publishes, drive /cmd_vel forward, confirm the
# reported /odom advances (the inverted-UART nav loop closes with no Gazebo).
# Prints `RIG2_RESULT: PASS|FAIL`. Run from the host with `make rig2`, or inside
# the autox container directly: `bash /autox/docker/rig2_mcu_bridge.sh`
# (the entrypoint has already sourced ROS + put the venv on PATH).
set -u
cd /autox

echo "[rig2] launching McuBridgeEngine (MCU=MOCK)…"
python run_mcu_bridge.py > /tmp/bridge.log 2>&1 &
BRIDGE_PID=$!

# Wait (<=12s) for /odom to be advertised.
for _ in $(seq 1 24); do
  ros2 topic list 2>/dev/null | grep -qx '/odom' && break
  sleep 0.5
done

echo "[rig2] nav topics present:"; ros2 topic list 2>/dev/null | grep -E '^/(odom|embedded_odom|cmd_vel)$' | sort

# --field still appends the '---' message separator; take only the value line.
X_BEFORE=$(timeout 6 ros2 topic echo --once --field pose.pose.position.x /odom 2>/dev/null | head -n1)
echo "[rig2] /odom position.x BEFORE = ${X_BEFORE:-<none>}"

echo "[rig2] driving /cmd_vel linear.x=0.5 for 3s…"
timeout 3 ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.5, y: 0.0, z: 0.0}}' >/dev/null 2>&1

X_AFTER=$(timeout 6 ros2 topic echo --once --field pose.pose.position.x /odom 2>/dev/null | head -n1)
echo "[rig2] /odom position.x AFTER  = ${X_AFTER:-<none>}"

kill "$BRIDGE_PID" 2>/dev/null
wait "$BRIDGE_PID" 2>/dev/null

echo "[rig2] bridge log (tail):"; tail -6 /tmp/bridge.log
python3 - "$X_BEFORE" "$X_AFTER" <<'PY'
import sys
b, a = sys.argv[1], sys.argv[2]
try:
    b, a = float(b), float(a)
except ValueError:
    print("RIG2_RESULT: FAIL (no /odom samples — bridge not publishing)"); sys.exit(0)
print(f"RIG2_RESULT: {'PASS' if a > b + 0.05 else 'FAIL'} (advanced {a-b:+.3f} m under /cmd_vel)")
PY
