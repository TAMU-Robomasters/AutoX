"""Check if competition software is running in the background as daemon process."""

import os

from src.toolbox.globals import path_to, print


def throw_if_autoboot_is_already_running():  # noqa: D103
    print("""Checking if autoboot is already running...""")
    if not os.path.isfile(path_to.autoboot_id):
        print("""nope""")
        return
    with open(path_to.autoboot_id, "r") as f:
        output = f.read()
        id_string = ""
        found_start = False
        for each in output:
            if not (found_start):
                if each == '"':
                    found_start = True
            elif each == '"':
                break
            else:
                id_string += each

        import subprocess

        stdout = subprocess.check_output(["ps", "aux"]).decode("utf-8")[0:-1]
        if id_string in stdout:
            raise Exception(
                """\n\n\nERROR: autoboot is already running\nRun this command to kill it:\n    ./commands/xavier/boot_control/kill_booted_process\nIf you try to kill it manually it will resurrect itself\n\n\n"""
            )
    print("""autoboot is not running ✅""")
