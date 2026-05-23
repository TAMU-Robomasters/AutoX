"""Singleton used to display OpenCV windows. Config decides if windows are shown."""

import cv2 as cv
import numpy as np

from src.subsystems.video_streaming.video_stream import video_stream


def _rgb(red, blue, green):
    """Helper to create RGB color tuples."""
    return (red, blue, green)


def _rgb_to_bgr(red, blue, green):
    """Convert RGB color tuple to BGR for OpenCV."""
    return (green, blue, red)


RED = _rgb(240, 113, 120)
CYAN = _rgb(137, 221, 255)
BLUE = _rgb(130, 170, 255)
GREEN = _rgb(4, 179, 39)
YELLOW = _rgb(254, 195, 85)


class Display:
    """Singleton to manage OpenCV windows for displaying images and visualizing data."""

    def __init__(self):
        """Initialize the Display singleton."""
        image = np.zeros((video_stream.height, video_stream.width, 3), dtype=np.uint8)
        self.windows = {"main": Window(image)}

    def add_window(self, name, image):
        """Add windows dynamically.

        Being able to add windows dynamically has some cons if you desire many windows
        (i.e. hard to track them all), but it is more flexible.
        Feel free to restructure display if needed.
        """
        self.windows[name] = Window(image)

    def show_windows(self):
        """Show the image in each window to the screen."""
        for name in self.windows:
            cv.imshow(name, self.windows[name].img)
            cv.waitKey(1)

    def __del__(self):
        """Destroy all OpenCV windows on deletion."""
        for name in getattr(self, "windows", {}):
            cv.destroyWindow(name)


class Window:
    """Container for an image with drawing utilities."""

    def __init__(self, image: np.ndarray):
        """OpenCV Window wrapper around an image."""
        self.img = image

    def add_point(self, *, x, y, color=YELLOW, radius=3):
        """Add a point to the image at the specified location."""
        color = _rgb_to_bgr(*color)
        self.img = cv.circle(
            self.img,
            (int(x), int(y)),
            radius,
            tuple(int(each) for each in color),
            thickness=-1,
            lineType=8,
            shift=0,
        )
        # TODO warn out of bounds

    def add_contour(self, contour, color=GREEN, thickness=2):
        """Add a contour to the image."""
        color = _rgb_to_bgr(*color)
        cv.drawContours(self.img, [contour], -1, color, thickness)

    def add_bounding_box(self, *, bounding_box, color=GREEN, thickness=2):
        """Add a bounding box to the image.

        @bounding_box:
            should be (x, y, width, height)
            the x,y should be the top-left corner of the image
            y=0 is the very top of the image
            x=0 is the left-most side of the image
        @color: tuple of RGB values, each are 0-255
        @thickness: int of how many pixels
        """
        color = _rgb_to_bgr(*color)
        if str(type(bounding_box)) == "<class 'toolbox.geometry_tools.BoundingBox'>":
            bounding_box = [
                bounding_box.x_top_left,
                bounding_box.y_top_left,
                bounding_box.width,
                bounding_box.height,
            ]

        # Starting cordinate
        start = (int(bounding_box[0]), int(bounding_box[1]))
        # Bottom right of the bounding box
        end = (
            int(bounding_box[0] + bounding_box[2]),
            int(bounding_box[1] + bounding_box[3]),
        )

        # Draw bounding box on image
        cv.rectangle(self.img, start, end, color, thickness)

    def add_text(self, *, text, location, color=(255, 255, 255), size=0.7):
        """Add text to the image at the specified location."""
        color = _rgb_to_bgr(*color)
        font = cv.FONT_HERSHEY_SIMPLEX
        line_type = 2
        cv.putText(self.img, text, location, font, size, color, line_type)


display = Display()
