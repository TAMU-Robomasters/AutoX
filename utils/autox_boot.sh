#!/usr/bin/env bash
#
# Wrapper run by the `autox_boot` systemd service on every boot.
# Runs `uv run main.py` in an infinite restart loop, logging to ~/boot.log.
#
# You normally don't run this by hand. Install it with:  ./utils/setup_boot_script.sh
# Remove it with:                                         ./utils/kill_onboot_cv
#
# Profiles/config come from src/local_data.ignore.yaml (selected_profiles), so the
# boot command needs no CLI args.
set -u

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
[ -x "$UV" ] || UV="$(command -v uv || echo uv)"

BOOT_LOG="$HOME/boot.log"
OLD_BOOT_LOG="$HOME/boot.old.log"

# rotate the previous boot log so we always have the last-two boots
rm -f "$OLD_BOOT_LOG"
[ -f "$BOOT_LOG" ] && cp "$BOOT_LOG" "$OLD_BOOT_LOG"
: > "$BOOT_LOG"

# send everything from here on to the boot log
exec >>"$BOOT_LOG" 2>&1

cd "$REPO_DIR" || { echo "ERROR: cannot cd to $REPO_DIR"; exit 1; }

while true; do
    echo "########## AUTOX BOOTING  $(date)  ##########"
    "$UV" run main.py
    echo "########## AUTOX PROCESS DIED — RESTARTING  $(date)  ##########"
    sleep 2
done
