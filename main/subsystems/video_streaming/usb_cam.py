import time
import itertools
# project imports
from toolbox.video_tools import Video
from toolbox.globals import path_to, config, print
import numpy as np

simulation = config.videostream.simulation

import cv2 as cv
import numpy as np
import threading
import queue

class BufferlesCvCapture:
    """
    Create thread to constantly read frames so that opencv doesn't buffer them.
    We want to always read the most recent frame.
    """
    def __init__(self, pipeline):
        self.cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise Exception("Could not open video.")
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame.")
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self) -> np.ndarray:
        return self.q.get()

class VideoStream:
    def __init__(self):
        # FIXME: probably don't hardcore this numbers @Jai
        pipeline = ("v4l2src device=/dev/video0 ! "
            "image/jpeg, width=1280, height=720, framerate=90/1 ! "
            "nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "appsink"
        ) 
        try:
            self.cap = BufferlesCvCapture(pipeline)
        except Exception as e:
            print(f"Error in BufferlesCvCapture: {e}")
            raise e
    
    def frames(self, non_threaded=False):
        yield self.cap.read()
        
    def save_video_if_needed(self):
        pass
    
    def get_intrinsics(self):
        distortion_matrix = np.load(f'{path_to.calibration_presets}/dist.pkl', allow_pickle=True)
        camera_matrix = np.load(f'{path_to.calibration_presets}/camera_matrix.pkl', allow_pickle=True)
        return distortion_matrix, camera_matrix
