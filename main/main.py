import atexit
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from toolbox.globals import config, runtime, time_synchronized, print
from toolbox.autoboot_check import throw_if_autoboot_is_already_running
from subsystems import video_stream, model, aim, communicate, log


if config.mode != "production":
    throw_if_autoboot_is_already_running()

atexit.register(log.when_iteration_stops)
# i.e. ctrl+C will trigger "when_iteration_stops"
# (its not perfectly reliable, but better than nothing)

# Run detection infinitely
for runtime.frame_number, runtime.color_image , runtime.depth_image in video_stream.frames():
    model.when_frame_arrives()
    aim.when_bounding_boxes_refresh()
    communicate.when_aiming_refreshes()
    log.when_finished_processing_frame()

log.when_iteration_stops()
