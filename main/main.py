import atexit
import sys
import os
import numpy as np
import cv2 as cv
from time import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from toolbox.globals import config, runtime, time_synchronized, print
from toolbox.autoboot_check import throw_if_autoboot_is_already_running
from subsystems import video_stream, communicate, log, armor, icon_detection, pnp, draw, frame_proccesing

if config.mode != "production":
    throw_if_autoboot_is_already_running()

atexit.register(log.when_iteration_stops)
# i.e. ctrl+C will trigger "when_iteration_stops"
# (its not perfectly reliable, but better than nothing)

# Run detection infinitely
for runtime.frame_number, runtime.color_image, runtime.depth_image in video_stream.frames():
    start_time = time()
    contours = frame_proccesing.frame_process(runtime.color_image)
    runtime.lights = armor.bounding_boxes(contours, runtime.color_image) 
    panels = None
    if len(runtime.lights) > 1:
        try:
            pairs = armor.pairing(runtime.lights)
            panels = []
            for pair in pairs:
                panel = armor.armour_corners(pair)
                good_panel = icon_detection.icon_detection(panel, runtime.color_image)
                if good_panel:
                    pnp.get_cord(panel)
                    if abs(np.degrees(panel.rvec[2])) < 40:
                        panels.append(panel)
                    else:#removes panels that are yawed to much
                        pass
                else:
                    pairs.remove(pair)
            if config.debug:
                print(f'''{len(panels)} panels found''', end=" ")
                for panel in panels:
                    draw.draw(panel, runtime.color_image)
        except Exception as e:
            raise e
            print(f"Error in get_cord: {e}")
            cords = []
    else:
        cords = []

    if config.debug:
        cv.resize(runtime.color_image, (640, 480))
        cv.imshow("runtime.color_image", runtime.color_image)
    
    end_time = time()
    elapsed_time = (end_time - start_time)
    
    communicate.when_aiming_refreshes(panels, start_time)
    log.when_finished_processing_frame()

log.when_iteration_stops()
