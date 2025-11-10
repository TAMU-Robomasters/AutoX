import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.ops import non_max_suppression

from toolbox.globals import print, print_synchronized, time_synchronized

class Yolov8(object):
    """
    description: A YOLOv8 class that wraps initialization, preprocess and postprocess ops.
    """
    
    def __init__(self, model_filepath, input_dimension, acceleration, warmup=True):
        # config check
        assert acceleration in ['tensor_rt', 'gpu', 'cpu', None ]
        self.acceleration = acceleration
        if acceleration in [ 'tensor_rt', 'gpu' ] and torch.cuda.is_available():
            print("[modeling]   gpu_acceleration: ENABLED\n")
            # loaded_model = loaded_model.to(torch.device("cuda"))
            self.device = torch.device("cuda")
            # print(next(self.model.parameters()).is_cuda)
        else:
            print("[modeling]   Running on CPU\n")
            self.device = torch.device("cpu")

        self.input_w = self.input_h = input_dimension

        if self.acceleration in [ None, "cpu", "gpu"]:
            if self.acceleration == 'gpu' and torch.cuda.is_available():
                print("[modeling]   gpu_acceleration: ENABLED\n")
                self.device = torch.device("cuda")
                overrides = {'model': model_filepath,
                        'device': "0",
                        'half': True,
                        'imgsz': f'{self.input_w},{self.input_h}'
                        }
            else:
                print("[modeling]   Running on CPU\n")
                self.device = torch.device("cpu")
                overrides = {'model': model_filepath,
                            'device': "cpu",
                            'imgsz': f'{self.input_w},{self.input_h}'
                            }
            # load model locally
            
            self.predictor = BasePredictor(overrides=overrides)

            self.predictor.setup_model(model=None)

            self.predict = lambda image_tens: self.predictor.model(image_tens)

            self.nms = lambda preds, *args, **kwargs: non_max_suppression(preds, *args, **kwargs)[0]

            if warmup:
                self.predictor.model.warmup(imgsz=(1, 3, self.input_h, self.input_w))
                print("\n[modeling]   Warmup complete\n")
        elif self.acceleration == 'tensor_rt':
            print("[modeling]   TensorRT: ENABLED\n")
            while True: # workaround for "Inconsistency detected by ld.so: dl-tls.c: 517: _dl_allocate_tls_init: Assertion `listp != NULL' failed!" error
                try:
                    import subsystems.modeling.plugins.yolo_v8_plugins.v8_inference_engine as trtengine
                except Exception as e:
                    print(e)
                    print("Trying again...")
                    continue
                break
            print("[modeling]   Loading TensorRT Engine...")
            self.device = torch.device("cuda")

            self.engine = trtengine.init(model_filepath, self.device)

            self.predict = lambda image_tens: trtengine.infer(image_tens, self.engine)
            
            # empty nms function because tensorrt does it internally
            self.nms = lambda preds, *args, **kwargs: preds
            print("[modeling]   Warning: TensorRT will use prebuilt NMS value of 0.2. To change, rebuild TensorRT engine with new value.\n")
        else:
            raise Exception(f'''acceleration == {acceleration}''')

    def preprocess_image(self, image_pre):
        # takes in (h, w, c) BGR image
        image_tensor = torch.as_tensor(image_pre, device=self.device) # create torch tensor directly on device (GPU or CPU)
        image_tensor = image_tensor.permute(2, 0, 1) # HWC to CHW
        image_tensor = image_tensor[[2, 1, 0]].unsqueeze(0) # RGB to BGR, add batch dimension
        image_tensor = F.interpolate(image_tensor, size=(self.input_w, self.input_h)).div(255.0) # resize, normalize from 0-255 to 0-1.0
        if self.acceleration == 'gpu':
            image_tensor = image_tensor.half()
        # WARNING: tensorrt doesn't support half precision, so we can't use .half() here
        return image_tensor, image_pre, image_pre.shape[0], image_pre.shape[1]

    def infer(self, image_tensor):
        return self.predict(image_tensor)

    def rescale_coords_to_original(self, boxes, image_post_dim):
        """
        description:    Rescale the coordinates of the boxes from model size to image size
        param:
            image_h:   height of original image
            image_w:   width of original image
            boxes:     A boxes numpy, each row is a box [x1, y1, x2, y2]
        return:
            boxes:     A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        h_rescaler = image_post_dim[0] / self.input_h
        w_rescaler = image_post_dim[1] / self.input_w

        boxes[:, [0, 2]] *= w_rescaler
        boxes[:, [1, 3]] *= h_rescaler
        return boxes

    def postprocess_preds(self, yolo_tens, image_post_h, image_post_w, conf_thres, detect_class=False):
        """
        description:    Postprocess the results from model inference
        param:
            yolo_tens:        A PyTorch Tensor, as output by YOLO
            image_post_h:     height of image to rescale box coords to
            image_post_w:     width of image to rescale box coords to
            detect_class:     Whether to detect class (in case model isn't trained to detect class)
        return:
            boxes:
        """
        preds = self.nms(yolo_tens, conf_thres=conf_thres, iou_thres=0.2)

        preds[:, :4] = self.rescale_coords_to_original(preds[:, :4], (image_post_h, image_post_w)).round()
        return preds[:, :4], preds[:, 4], preds[:, 5]
