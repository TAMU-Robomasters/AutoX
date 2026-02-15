# library imports
import os

from toolbox.geometry_tools import BoundingBox

# project imports
from toolbox.globals import config, path_to, print

#
# config
#
hardware_acceleration = config.model.hardware_acceleration
input_dimension = config.model.input_dimension
which_model = config.model.which_model

#
#
# helpers
#
#
def init_yolo_v8(model):
    print("\n[modeling]   YOLO v8 loading")

    from subsystems.modeling.plugins.yolo_v8_plugins.yolo_v8_plugin import Yolov8

    checked_path = None
    if hardware_acceleration == 'cpu' or hardware_acceleration == 'gpu':
        if not os.path.exists(path_to.yolo_v8.pytorch_model):
            raise Exception(
                f"model not found at {path_to.yolo_v8.pytorch_model}")
        checked_path = path_to.yolo_v8.pytorch_model
    elif hardware_acceleration == 'tensor_rt':
        if not os.path.exists(path_to.yolo_v8.tensor_rt_engine):
            raise Exception(
                f"model not found at {path_to.yolo_v8.tensor_rt_engine}")
        checked_path = path_to.yolo_v8.tensor_rt_engine

    # create Yolov8 object
    model.yolov8 = Yolov8(model_filepath=checked_path,
                        input_dimension=config.model.input_dimension,
                        acceleration=config.model.hardware_acceleration)

    # export data
    model.get_bounding_boxes = lambda *args, **kwargs: yolo_v8_beta_bounding_boxes(
        model, *args, **kwargs)

def yolo_v8_beta_bounding_boxes(model, frame, minimum_confidence):
    # t0 = time_synchronized()

    image_tensor, image_pre, image_pre_h, image_pre_w = model.yolov8.preprocess_image(frame)

    # t1 = time_synchronized()

    yolo_tens = model.yolov8.infer(image_tensor)

    # t2 = time_synchronized()

    raw_boxes, confidences, class_ids = model.yolov8.postprocess_preds(yolo_tens, image_pre_h, image_pre_w, minimum_confidence, detect_class=False)

    # t3 = time_synchronized()

    boxes = [BoundingBox.from_points(top_left=(x1, y1), bottom_right=(x2, y2)) for x1, y1, x2, y2 in raw_boxes]

    # t4 = time_synchronized()

    # print(f"\n\npreprocess: {t1 - t0:.3f} s")
    # print(f"inference: {t2 - t1:.3f} s")
    # print(f"postprocess: {t3 - t2:.3f} s")
    return boxes, confidences, class_ids