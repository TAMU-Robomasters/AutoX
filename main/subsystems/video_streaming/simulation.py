import time
import itertools
# project imports
from toolbox.video_tools import Video
from toolbox.globals import path_to, config, print
from toolbox.video_tools import Video
from toolbox.globals import path_to, config, print
import numpy as np

simulation = config.videostream.simulation

class VideoStream:
    def __init__(self):
        self.video_object = Video(path=simulation.input_file)
        
        if simulation.grab_method == 'threaded_frame':
            self.threadhreaded_object = CameraThreader(self, update_rate=simulation.threaded_update_rate)
        elif simulation.grab_method == 'next_frame':
            pass # no special setup but demo the case for consistency and the error message
        elif simulation.grab_method == 'latest_frame':
            print('Loading all frames into ram for simulated testing')
            self.all_frames = tuple(self.video_object.frames())
            print(f'Found {len(self.all_frames)} frames')
            self.start_time = None
            self.frame_rate = simulation.assumed_framerate
        else:
            raise Exception(f'simulated VideoStream was created, but config.videostream.simulation.grab_frame was {simulation.grab_frame} instead of one of ["next_frame", "latest_frame"]')
    
    def frames(self, non_threaded=False):
        if not non_threaded and simulation.grab_method == 'threaded_frame':
            while not self.threadhreaded_object.stopped:
                yield self.threadhreaded_object.frame # NOTE: has the slight possibility of sending the same frame twice
        else:
            # for now it simply doesn't exist
            depth_frame = None
            
            # just pass along the frames
            if simulation.grab_method == 'next_frame' or simulation.grab_method == 'threaded_frame':
                frame_number = 0
                for color_frame in self.video_object.frames():
                    frame_number += 1
                    yield frame_number, color_frame, depth_frame
            elif simulation.grab_method == 'latest_frame':
                self.start_time = time.time()
                for frame_number in itertools.count(1):
                    # figure out which frame should be retrieved based on the elapsed time
                    seconds_since_start = time.time() - self.start_time 
                    which_frame_index = int(seconds_since_start * self.frame_rate)
                    # stop if too much time has passed
                    if which_frame_index >= len(self.all_frames):
                        break
                    
                    yield frame_number, self.all_frames[which_frame_index], depth_frame
            else:
                raise Exception(f'''Unknown config.videostream.simulation.grab_method: {config.videostream.simulation.grab_method}, expected one of: [ 'next_frame', 'latest_frame', 'threaded_frame' ]''')
        
    def save_video_if_needed(self):
        pass


from time import sleep
class CameraThreader:
    def __init__(self, video_stream, update_rate=0.001):
        from threading import Thread
        self.video_stream = video_stream
        self.update_rate = update_rate
        
        self.frame = None

        self.stopped = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        
        self.start()

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        for self.frame in self.video_stream.frames(non_threaded=True):
            sleep(self.update_rate)
        self.stopped = True
        
    def stop(self):
        self.stopped = True

    def __del__(self):
        self.thread.join()