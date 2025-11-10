# library imports
import numpy as np
import cv2
import os
import time
from super_map import LazyDict

from math import dist, sqrt

# project imports
from toolbox.globals import path_to, config, print, runtime
from toolbox.geometry_tools import BoundingBox, Position
from toolbox.image_tools import Image
import subsystems.aim as aiming

# 
# config
# 
our_team_color        = config.our_team_color
hardware_acceleration = config.model.hardware_acceleration
input_dimension       = config.model.input_dimension
which_model           = config.model.which_model

# config check
assert hardware_acceleration in ['tensor_rt', 'gpu', 'cpu',]

# 
# shared data (imported by aiming and integration)
# 
runtime.modeling = LazyDict(
    bounding_boxes=[],
    enemy_boxes=[],
    confidences=[],
    best_bounding_box=[],
    current_confidence=0,
    found_robot=False,
)

# 
# 
# load model
# 
# 
model = LazyDict(
    W=None,
    H=None,
)

if which_model == 'yolo_v8':
    from subsystems.modeling.yolo_v8 import init_yolo_v8
    init_yolo_v8(model)
else:
    raise Exception("Model specified under /'model.which_model/' is not supported")

np.seterr(all='raise')

# 
# 
# main function
# 
# 
def when_frame_arrives():
    frame = runtime.color_image
    
    # 
    # all boxes
    # 
    all_boxes, confidences, class_ids = model.get_bounding_boxes(
        frame=frame,
        minimum_confidence=config.model.minimum_confidence,
    )
    
    screen_center = compute_screen_center(frame)
    # print("Screen center: " + str(screen_center))
    # 
    # remove our team
    # 
    if config.filter_team_color:
        enemy_boxes, confidences, class_ids = filter_plate_color(all_boxes, confidences, class_ids, our_team_color)
    
    # best box
    # best_bounding_box, current_confidence = get_optimal_bounding_box(
    #     boxes=enemy_boxes,
    #     confidences=confidences,
    #     screen_center=screen_center,
    # )
    
    # export data
    runtime.screen_center               = screen_center
    runtime.modeling.bounding_boxes     = all_boxes
    runtime.modeling.enemy_boxes        = enemy_boxes
    runtime.modeling.confidences        = confidences
    # runtime.modeling.best_bounding_box  = best_bounding_box
    # runtime.modeling.current_confidence = current_confidence
    # runtime.modeling.found_robot        = best_bounding_box is not None

# 
# 
# helpers
# 
# 
def get_optimal_bounding_box(boxes, confidences, screen_center):
    """
    Decide the single best bounding box to aim at using a score system.

    Input: All detected bounding boxes with their confidences and the screen_center location of the image.
    Output: Best bounding box and its confidence.
    """
    # no boxes
    if not boxes:
        return None, 0
    # if len(boxes) == 1:
    #     return boxes[0], confidences[0]

    best_bounding_box = boxes[0]
    best_score = 0
    best_conf = 0

    screen_center_normalizer = dist((screen_center[0]*2,screen_center[1]*2),(screen_center[0],screen_center[1])) # Find constant used to scale distance part of score to 1
    size_normalizer = 0.7 # plate at closest distance is 0.7 of the screen

    # Sequentially iterate through all bounding boxes
    for conf, box in zip(confidences, boxes):
        size_score = ((box.width / (runtime.color_image.shape[1])) / size_normalizer) # Compute score using size of box, relative to total image size
        print(f"size_score: {size_score}")
        center_score = (1 - dist(screen_center,(box[0] + box[2]/2, box[1] + box[3]/2)) / screen_center_normalizer) # scaled to 1
        print(f"center_score: {center_score}")
        conf_score = conf**2 # Compute score using confidence
        print(f"conf_score: {conf_score}")
        score = 0.75 * size_score + 0.125 * center_score + 0.125 * conf_score # Compute score using weighted average
        print(f"score: {score}")

        # Make current box the best if its score is the best so far
        if score > best_score:
            best_bounding_box = box
            best_conf = conf
            best_score = score
    # if best_score < 0.15:
    #     return None, 0
    # if size_score < 5:
    #     return None, 0
    return best_bounding_box, best_conf

color_to_class_id = dict(
    red=0,
    blue=1,
)

def filter_plate_color(boxes, confidences, class_ids, color_to_remove):
    """
    Filter bounding boxes based on team color.

    Input: Zipped model result of boxes, confidences, and class_ids
    Output: Filtered boxes, confidences, and class_ids
    """
    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []

    # only do this if there are boxes
    if boxes:
        filtered_data = [(box, conf, class_id) for box, conf, class_id in zip(boxes, confidences, class_ids) if class_id == color_to_class_id[color_to_remove]]
        # only unzip if there are boxes left after filtering
        if filtered_data:
            filtered_boxes, filtered_confidences, filtered_class_ids = zip(*filtered_data)
    return filtered_boxes, filtered_confidences, filtered_class_ids

def compute_screen_center(color_image):
    # print("Color image shape: " + str(color_image.shape))
    return (color_image.shape[1] // 2, color_image.shape[0] // 2)