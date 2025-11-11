import cv2 as cv
import numpy as np

from toolbox.globals import config, runtime, time_synchronized, print

def frame_process(frame):
    """
    function will split color channels, threshold, and find contours.
    :param frame: input frame
    :param enemy_color: color of the enemy
    :return: contours
    """

    if config.our_team_color == 'blue':
        _, thresh = cv.threshold(frame[:,:,0], config.classical.blue_thresh[0], config.classical.blue_thresh[1], cv.THRESH_BINARY) # tune before match
    elif config.our_team_color == 'red':
        _, thresh = cv.threshold(frame[:,:,2], config.classical.red_thresh[0], config.classical.red_thresh[1], cv.THRESH_BINARY) # tune before match
    else:
        print('invalid color')

    kernel = np.ones((3,3),np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return contours