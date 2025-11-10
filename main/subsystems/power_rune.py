import math
import collections
import numpy as np

from toolbox.globals import runtime, path_to
from toolbox.image_tools import Image, rgb
from toolbox.geometry_tools import BoundingBox


def find_aiming_point():
    runtime.color_image
    
    # Contours
        # TODO
    
    # SubContours
        # TODO
    
    # Bounding Box Rect
        # TODO
    
    if debug:
        image = Image(runtime.color_image)
        top_left_corner_x = 0
        top_left_corner_y = 0
        box_width = 50
        box_height = 50
        image.add_bounding_box([
            top_left_corner_x,
            top_left_corner_y,
            box_width,
            box_height,
        ], color=rgb(255, 255, 255))
        image.show_and_pause()

# 
# test the function above
# 
print("imported")
if __name__ == '__main__':
    debug = True
    runtime.color_image = Image(path_to.rune_example).img # numpy array
    print("running")
    find_aiming_point()