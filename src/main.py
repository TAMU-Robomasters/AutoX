"""Main thing for real."""

#! check if this has multi threading
from src.toolbox.autoboot_check import throw_if_autoboot_is_already_running
from src.toolbox.globals import config
from src.engines.rerun import RerunEngine
useSimluation = True

def main():
    """Main function to run the armor detection and processing loop."""

    simulation: RerunEngine = RerunEngine()
    if(useSimluation == True):
        simulation.start()

