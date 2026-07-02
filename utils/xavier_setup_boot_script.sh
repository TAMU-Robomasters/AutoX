#!/usr/bin/env bash
#
# Set up AutoX to run on boot via systemd.
#
# What this does:
#   1. Generates an *editable* boot command file (utils/boot_command.ignore.sh)
#      whose only job is to `exec uv run main.py`. Profiles come from
#      src/local_data.ignore.yaml (selected_profiles) -- edit that file, or add
#      @PROFILE flags to the boot command, to change what runs.
#   2. Installs a systemd service (autox.service) that runs that command as the
#      current user, with Restart=always, on every boot.
#   3. Enables + starts the service.
#
# Replaces the old Deno-based xavier_setup_boot_script.js (Deno is not installed
# and its main/main.py path + REALSENSE profile are gone).
#
# Run it as your normal user (it will sudo for the privileged bits):
#   ./utils/xavier_setup_boot_script.sh
#
# To stop it from booting again, run:  ./utils/xavier_kill_onboot_cv

set -euo pipefail

# --- sanity: not root, is Linux/Jetson -------------------------------------
if [ "$(id -u)" = "0" ]; then
    echo "Don't run this as root -- run it as your normal user; it will sudo when needed." >&2
    exit 1
fi
if [ "$(uname -s)" != "Linux" ]; then
    echo "This script is for the Jetson (Linux) only. Do not run it on your PC." >&2
    exit 1
fi

# --- resolve paths dynamically (no hardcoded /home/orin/main/main.py) -------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
MAIN_PY="$REPO/main.py"
UV="$(command -v uv || true)"
RUN_USER="$USER"
RUN_HOME="$HOME"
BOOT_CMD="$SCRIPT_DIR/boot_command.ignore.sh"   # editable, generated once
BOOT_LOG="$RUN_HOME/boot.log"
SERVICE_NAME="autox"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

[ -f "$MAIN_PY" ] || { echo "Cannot find main.py at $MAIN_PY" >&2; exit 1; }
[ -n "$UV" ]      || { echo "Cannot find 'uv' on PATH. Install uv first." >&2; exit 1; }

# PATH the service needs: uv (~/.local/bin) + cuda + the usual system dirs.
UV_BIN_DIR="$(dirname "$UV")"
SERVICE_PATH="${UV_BIN_DIR}:/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

echo "repo:    $REPO"
echo "uv:      $UV"
echo "user:    $RUN_USER  (home: $RUN_HOME)"
echo "service: $SERVICE_NAME -> $UNIT_PATH"
echo

# --- 1. editable boot command (generate once, never clobber user edits) ----
if [ ! -f "$BOOT_CMD" ]; then
    echo "Generating editable boot command: $BOOT_CMD"
    cat > "$BOOT_CMD" <<'EOF'
#!/usr/bin/env bash
#
# EDIT ME. This is the command AutoX runs on boot.
#
# Profiles are read from src/local_data.ignore.yaml (selected_profiles), so the
# bare command is all you need. To override per-boot, append flags, e.g.:
#     exec uv run main.py @BOARD=XAVIER @GPU=TENSOR_RT
#
exec uv run main.py
EOF
    chmod +x "$BOOT_CMD"
else
    echo "Keeping existing boot command (edit it yourself): $BOOT_CMD"
    chmod +x "$BOOT_CMD"
fi

touch "$BOOT_LOG"

# --- 2. systemd unit --------------------------------------------------------
# User=$RUN_USER so serial (/dev/ttyTHS1, dialout) + camera (/dev/video*, video)
# work without root; systemd initializes the user's supplementary groups, and we
# name dialout/video explicitly as belt-and-suspenders. Restart=always replaces
# the old hand-rolled `while true` loop.
UNIT_CONTENT="[Unit]
Description=AutoX auto-aim (uv run main.py) on boot
After=network.target

[Service]
Type=simple
User=${RUN_USER}
SupplementaryGroups=dialout video i2c gpio render
WorkingDirectory=${REPO}
Environment=HOME=${RUN_HOME}
Environment=PATH=${SERVICE_PATH}
ExecStart=${BOOT_CMD}
Restart=always
RestartSec=2
StandardOutput=append:${BOOT_LOG}
StandardError=append:${BOOT_LOG}

[Install]
WantedBy=multi-user.target
"

echo "Installing service (needs sudo)..."
printf '%s' "$UNIT_CONTENT" | sudo tee "$UNIT_PATH" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo
echo "Done. AutoX will now start on boot and restart if it dies."
echo "    boot command: $BOOT_CMD   (edit this to change flags)"
echo "    boot log:     $BOOT_LOG"
echo "    live logs:    journalctl -u $SERVICE_NAME -f"
echo "    stop booting: $SCRIPT_DIR/xavier_kill_onboot_cv"
