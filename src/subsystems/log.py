"""All logging for cv's auto-aim system is done here.
"""
from datetime import datetime as dt

from super_map import LazyDict

from toolbox.globals import config, runtime
from toolbox.image_tools import Image, _rgb

#
# config
#
display_live_frames       = config.log.display_live_frames
camera                    = config.hardware.camera
depth_compatible          = config.hardware.camera_has_depth
should_benchmark          = config.mode == 'benchmark'

# create incremented storage path
timestamp = dt.now()



def generate_image(fps=0):
    color_image        = runtime.color_image
    found_robot        = runtime.get("found_robot", False)
    current_confidence = runtime.get("current_confidence", False)
    best_bounding_box  = runtime.get("best_bounding_box", None)
    bounding_boxes     = [
        # should be: x_top_left, y_top_left, width, height format
        # but might not be (cx sounds like center x)
        (each.cx, each.cy, each.w, each.h)
            for each in runtime.lights
    ]
    enemy_boxes        = runtime.get("enemy_boxes", [])
    center_point       = runtime.get("aiming",{}).get("center_point",None)
    target_3d          = runtime.get("aiming",{}).get("target_3d",None)
    status             = runtime.get("aiming",{}).get("target_status",LazyDict(name=""))
    
    image = Image(runtime.color_image)

    if len(bounding_boxes) > 0:
        white  = _rgb(255, 255, 255)
        red    = _rgb(240, 113, 120)
        blue   = _rgb(130, 170, 255)
        cyan   = _rgb(137, 221, 255)
        green  = _rgb(195, 232, 141)
        yellow = _rgb(254, 195,  85)
        for each in bounding_boxes:
            # print(f"visual bounding_box: {each}")
            image.add_bounding_box(each, color=_rgb(255, 255, 255))
        for each in enemy_boxes:
            image.add_bounding_box(each, color=_rgb(254, 195,  85))
        if found_robot:
            image.add_bounding_box(best_bounding_box, color=_rgb(240, 113, 120))
            image.add_point(x=center_point.x     , y=center_point.y     , color=_rgb(130, 170, 255), radius=10)
            # image.add_point(x=prediction_point.x , y=prediction_point.y , color=rgb(195, 232, 141), radius=5)
    
    x_location = 30
    y_location = 50
    if depth_compatible:
        disp_target_3d = [round(x, 3) for x in target_3d] if target_3d else ["NAN, NAN, NAN"]
        image.add_text(text=f"target_3d: {    disp_target_3d         }", location=(x_location, y_location)); y_location += 50
    image.add_text(text=f"confidence: {       current_confidence :.2f}", location=(x_location, y_location)); y_location += 50
    image.add_text(text=f"status: {           status.name            }", location=(x_location, y_location)); y_location += 50
    image.add_text(text=f"fps: {              fps                :.2f}", location=(x_location, y_location)); y_location += 50
        
    return image
