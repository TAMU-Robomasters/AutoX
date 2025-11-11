from toolbox.globals import path_to, config, print
from toolbox.image_tools import Image

# 
# select which file to import from
# 
if config.hardware.camera == 'zed':
    from subsystems.video_streaming.zed import VideoStream
elif config.hardware.camera == 'realsense':
    from subsystems.video_streaming.realsense import VideoStream
elif config.hardware.camera == 'usb_cam':
    from subsystems.video_streaming.usb_cam import VideoStream
else:
    from subsystems.video_streaming.simulation import VideoStream

video_stream = VideoStream()
frames = video_stream.frames