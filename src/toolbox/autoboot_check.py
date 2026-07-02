"""Refuse to start a manual AutoX instance while the autoboot service owns the hardware.

The robot autostarts AutoX via the ``autox_boot`` systemd service (installed by
``utils/setup_boot_script.sh``), which runs ``uv run main.py`` in a restart loop. That
instance owns the camera and its iceoryx2 publisher, so launching a *second*
``uv run main.py`` by hand collides on the shared-memory publisher and dies with a
cryptic ``ExceedsMaxSupportedPublishers`` (and shows no frames). This guard detects the
running service and fails fast with instructions instead.

Replaces the old ``run/autoboot_id`` pid-file check, which no longer matches the
systemd-based autoboot and so silently did nothing.
"""

import subprocess

from src.toolbox.logger import get_logger

log = get_logger("autoboot_check")

# systemd unit installed by utils/setup_boot_script.sh (utils/kill_onboot_cv removes it).
_SERVICE = "autox_boot.service"


def _running_under_autoboot_service() -> bool:
    """Return True if *this* process was launched by the autoboot service.

    systemd places the whole service (wrapper -> uv -> python -> engine children) in the
    unit's cgroup, so the boot-launched instance sees the unit name in
    ``/proc/self/cgroup``. Used so the service never blocks itself.
    """
    try:
        with open("/proc/self/cgroup", "r") as f:
            return _SERVICE in f.read()
    except OSError:
        return False


def _autoboot_service_is_active() -> bool:
    """Return True if the autoboot systemd service is currently active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", _SERVICE],
            check=False,
        )
    except OSError:
        # no systemd / systemctl (e.g. a laptop dev box) -> no autoboot to collide with
        return False
    return result.returncode == 0


def throw_if_autoboot_is_already_running() -> None:
    """Fail fast if the autoboot service already owns the hardware.

    No-op when we *are* the autoboot instance, or when the service isn't running.
    """
    log.info("Checking if autoboot service (%s) is running...", _SERVICE)

    if _running_under_autoboot_service():
        log.info("This process is the autoboot instance — continuing.")
        return

    if not _autoboot_service_is_active():
        log.info("autoboot is not running ✅")
        return

    raise RuntimeError(
        "\n\n\n"
        f"ERROR: the '{_SERVICE}' autoboot service is already running and owns the\n"
        "camera + iceoryx2 publisher. A second 'uv run main.py' will collide on the\n"
        "shared-memory publisher (ExceedsMaxSupportedPublishers) and see no frames.\n\n"
        "  Stop it for this session (restarts on reboot):\n"
        "      sudo systemctl stop autox_boot.service\n\n"
        "  Disable autoboot permanently:\n"
        "      ./utils/kill_onboot_cv\n"
        "\n\n"
    )
