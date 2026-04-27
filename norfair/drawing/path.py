from collections import defaultdict
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from norfair.drawing.color import Palette
from norfair.drawing.drawer import Drawer
from norfair.tracker import TrackedObject
from norfair.utils import warn_once


class Paths:
    """
    Class that draws the paths taken by a set of points of interest defined from the coordinates of each tracker estimation.

    Parameters
    ----------
    get_points_to_draw : Optional[Callable[[np.array], np.array]], optional
        Function that takes a list of points (the `.estimate` attribute of a [`TrackedObject`][norfair.tracker.TrackedObject])
        and returns a list of points for which we want to draw their paths.

        By default it is the mean point of all the points in the tracker.
    thickness : Optional[int], optional
        Thickness of the circles representing the paths of interest.
    color : Optional[Tuple[int, int, int]], optional
        [Color][norfair.drawing.Color] of the circles representing the paths of interest.
    radius : Optional[int], optional
        Radius of the circles representing the paths of interest.
    attenuation : float, optional
        A float number in [0, 1] that dictates the speed at which the path is erased.
        if it is `0` then the path is never erased.

    Examples
    --------
    >>> from norfair import Tracker, Video, Path
    >>> video = Video("video.mp4")
    >>> tracker = Tracker(...)
    >>> path_drawer = Path()
    >>> for frame in video:
    >>>    detections = get_detections(frame)  # runs detector and returns Detections
    >>>    tracked_objects = tracker.update(detections)
    >>>    frame = path_drawer.draw(frame, tracked_objects)
    >>>    video.write(frame)
    """

    def __init__(
        self,
        get_points_to_draw: Optional[Callable[[np.array], np.array]] = None,
        thickness: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
        radius: Optional[int] = None,
        attenuation: float = 0.01,
    ):
        if get_points_to_draw is None:

            def get_points_to_draw(points):
                pass

        self.get_points_to_draw = get_points_to_draw

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.mask = None
        self.attenuation_factor = 1 - attenuation

    def draw(
        self, frame: np.ndarray, tracked_objects: Sequence[TrackedObject]
    ) -> np.array:
        """
        Draw the paths of the points interest on a frame.

        !!! warning
            This method does **not** draw frames in place as other drawers do, the resulting frame is returned.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on.
        tracked_objects : Sequence[TrackedObject]
            List of [`TrackedObject`][norfair.tracker.TrackedObject] to get the points of interest in order to update the paths.

        Returns
        -------
        np.array
            The resulting frame.
        """
        pass


class AbsolutePaths:
    """
    Class that draws the absolute paths taken by a set of points.

    Works just like [`Paths`][norfair.drawing.Paths] but supports camera motion.

    !!! warning
        This drawer is not optimized so it can be stremely slow. Performance degrades linearly with
        `max_history * number_of_tracked_objects`.

    Parameters
    ----------
    get_points_to_draw : Optional[Callable[[np.array], np.array]], optional
        Function that takes a list of points (the `.estimate` attribute of a [`TrackedObject`][norfair.tracker.TrackedObject])
        and returns a list of points for which we want to draw their paths.

        By default it is the mean point of all the points in the tracker.
    thickness : Optional[int], optional
        Thickness of the circles representing the paths of interest.
    color : Optional[Tuple[int, int, int]], optional
        [Color][norfair.drawing.Color] of the circles representing the paths of interest.
    radius : Optional[int], optional
        Radius of the circles representing the paths of interest.
    max_history : int, optional
        Number of past points to include in the path. High values make the drawing slower

    Examples
    --------
    >>> from norfair import Tracker, Video, Path
    >>> video = Video("video.mp4")
    >>> tracker = Tracker(...)
    >>> path_drawer = Path()
    >>> for frame in video:
    >>>    detections = get_detections(frame)  # runs detector and returns Detections
    >>>    tracked_objects = tracker.update(detections)
    >>>    frame = path_drawer.draw(frame, tracked_objects)
    >>>    video.write(frame)
    """

    def __init__(
        self,
        get_points_to_draw: Optional[Callable[[np.array], np.array]] = None,
        thickness: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
        radius: Optional[int] = None,
        max_history=20,
    ):

        if get_points_to_draw is None:

            def get_points_to_draw(points):
                pass

        self.get_points_to_draw = get_points_to_draw

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.past_points = defaultdict(lambda: [])
        self.max_history = max_history
        self.alphas = np.linspace(0.99, 0.01, max_history)

    def draw(self, frame, tracked_objects, coord_transform=None):
        pass
