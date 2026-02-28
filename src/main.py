"""Main thing for real."""

#! check if this has multi threading
from src.toolbox.autoboot_check import throw_if_autoboot_is_already_running
from src.toolbox.globals import config


def main():
    """Main function to run the armor detection and processing loop."""
    if config.mode != "production":
        throw_if_autoboot_is_already_running()
   
    from src.engines.circlet import CircletEngine
    circlet = CircletEngine()
    circlet.start()
    
