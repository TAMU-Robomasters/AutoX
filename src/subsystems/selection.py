from numpy.distutils.command.config import config
from src.core.module import Module
from src.pipelines.aim.context import AutoAimContext
from src.subsystems.video_streaming.video_stream import video_stream
from math import dist, exp
from src.toolbox.globals import config
from src.toolbox.geometry_tools import Position

MIN_RANGE = config.aiming.max_range

class SelectingWith3DModule(Module[AutoAimContext]):
    """Selecting module that used 3D information to select the closest target."""
    def __init__(self):
        """Initialize the SelectingWith3DModule."""
        super().__init__(
            name="SelectingWith3DModule",
            inputs=["panels"],
            outputs=["target_panel"],
        )

    def run(self, ctx: AutoAimContext) -> AutoAimContext:
        boxes = [panel.bbx for panel in ctx.panels] if ctx.panels else []
        valid3dTargets = [panel.position for panel in ctx.panels] if ctx.panels else []
        best_targ_3d = None
        best_score = 0
        best_idx = None
    

        screen_center = video_stream.center

        screen_center_normalizer = dist((screen_center[0]*2,screen_center[1]*2),(screen_center[0],screen_center[1])) # Find constant used to scale distance part of score to 1
        size_normalizer = 0.7 # plate at closest distance is 0.7 of the screen

        # Sequentially iterate through all bounding boxes
        for idx, (box, targetXYZ) in enumerate(zip(boxes, valid3dTargets)):
            size_score = ((box.width / (video_stream.width)) / size_normalizer) # Compute score using size of box, relative to total image size
            print(f"size_score: {size_score}")
            center_score = (1 - dist(screen_center,(box[0] + box[2]/2, box[1] + box[3]/2)) / screen_center_normalizer) # scaled to 1
            print(f"center_score: {center_score}")

            # clamped to 0 to 1
            # this is a 2d point - want to draw a circle
            # radius dependent on the depth? tweak1
            # exponential instead of linear? tweak2
            
            previous_panel_center = video_stream.center if (ctx.prev_target_panel is None) else ctx.prev_target_panel.bbx.center # in pixels  # ty:ignore[possibly-missing-attribute]
            #TODO: ? do we want to keep position class
            distance = (dist(previous_panel_center, Position(box.center)) / 100)
            radius = 100
            if distance >= radius:
                circle_bias_score = 0  # The point is at the edge or outside the circle
            else:
                circle_bias_score = 1 - (distance / radius)  # Calculate the score based on the normalized distance
            print(f"circle_bias_score: {circle_bias_score}")
            
            # linear appraoch (based on max and min range in info.yaml min and max is 1m to 5m)
            # ex. when targetXYZ[1] is 3.5, 
            # the depth_score is computed as 0.625, which falls within the clamped range of 0 to 1.
            # depth_score = max(0, min(1, (targetXYZ[1] - 1.0) / 4.0))

            # exponential approach 
            # (values closer to 1 will result in higher depth_score values)
            # ex. targetXYZ[1] is 3.5, the depth_score calculated using the exponential approach is approximately 0.2865
            depth_score = max(0, min(MIN_RANGE, exp((1 - targetXYZ[1]) / 2)))
            print(f"depth_score: {depth_score}")

            # Compute score using weighted average
            score = 0.125 * size_score + 0.125 * center_score + 0.125 * depth_score + 0.625 * circle_bias_score 

            # Make current box the best if its score is the best so far
            if score > best_score:
                best_targ_3d = targetXYZ
                best_score = score
                best_idx = idx
            
        ctx.target_panel = None if best_targ_3d is None else ctx.panels[best_idx] # ty:ignore[assignment, invalid-argument-type, not-subscriptable]
        return ctx