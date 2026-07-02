#!/usr/bin/env bash
#
# Install + enable the AutoX autoboot systemd service.
#
# On boot the service runs `uv run main.py` in an infinite restart loop (see
# autox_boot.sh), using the profiles saved in src/local_data.ignore.yaml.
#
#   Set up autoboot:  ./utils/setup_boot_script.sh
#   Remove autoboot:  ./utils/kill_onboot_cv
#
# Safe to re-run: it just re-installs the unit with current paths.
set -euo pipefail

SERVICE_NAME="autox_boot"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WRAPPER="$SCRIPT_DIR/autox_boot.sh"

# Run the app as the invoking (non-root) user, not root — uv, the venv and the
# X display all live in that user's home.
RUN_USER="${SUDO_USER:-$USER}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

# sanity checks
[ -f "$REPO_DIR/main.py" ] || { echo "ERROR: $REPO_DIR/main.py not found"; exit 1; }
[ -f "$WRAPPER" ]         || { echo "ERROR: wrapper $WRAPPER not found"; exit 1; }
chmod +x "$WRAPPER"

# A previous root/sudo run may have left the boot logs root-owned, which the
# service (running as $RUN_USER) can't truncate. Clear them so the wrapper can
# recreate them as the right user.
sudo rm -f "$RUN_HOME/boot.log" "$RUN_HOME/boot.old.log"

UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
echo "Installing $UNIT_PATH"
echo "  user: $RUN_USER    repo: $REPO_DIR"

sudo tee "$UNIT_PATH" >/dev/null <<EOF
[Unit]
Description=AutoX CV autoboot (uv run main.py)
# NOTE: do not add graphical.target here. It is ordered *after* multi-user.target,
# and this unit is WantedBy=multi-user.target, so an After=graphical.target creates
# an ordering cycle and systemd silently deletes our start job at boot.
After=network.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO_DIR
Environment=HOME=$RUN_HOME
Environment=PATH=$RUN_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
ExecStart=$WRAPPER
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

cat <<EOF

Done. AutoX will now start on boot.
  status:   systemctl status $SERVICE_NAME
  logs:     tail -f $RUN_HOME/boot.log      (or: journalctl -u $SERVICE_NAME -f)
  disable:  $SCRIPT_DIR/kill_onboot_cv
EOF
