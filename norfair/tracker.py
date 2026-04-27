from logging import warning
from typing import Any, Callable, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
from rich import print

from norfair.camera_motion import CoordinatesTransformation

from .distances import (
    AVAILABLE_VECTORIZED_DISTANCES,
    ScalarDistance,
    get_distance_by_name,
)
from .filter import FilterFactory, OptimizedKalmanFilterFactory
from .utils import validate_points


class Tracker:
    """
    The class in charge of performing the tracking of the detections produced by a detector.

    Parameters
    ----------
    distance_function : Union[str, Callable[[Detection, TrackedObject], float]]
        Function used by the tracker to determine the distance between newly detected objects and the objects that are currently being tracked.
        This function should take 2 input arguments, the first being a [Detection][norfair.tracker.Detection], and the second a [TrackedObject][norfair.tracker.TrackedObject].
        It has to return a `float` with the distance it calculates.
        Some common distances are implemented in [distances][], as a shortcut the tracker accepts the name of these [predefined distances][norfair.distances.get_distance_by_name].
        Scipy's predefined distances are also accepted. A `str` with one of the available metrics in
        [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
    distance_threshold : float
        Defines what is the maximum distance that can constitute a match.
        Detections and tracked objects whose distances are above this threshold won't be matched by the tracker.
    hit_counter_max : int, optional
        Each tracked objects keeps an internal hit counter which tracks how often it's getting matched to a detection,
        each time it gets a match this counter goes up, and each time it doesn't it goes down.

        If it goes below 0 the object gets destroyed. This argument defines how large this inertia can grow,
        and therefore defines how long an object can live without getting matched to any detections, before it is displaced as a dead object, if no ReID distance function is implemented it will be destroyed.
    initialization_delay : Optional[int], optional
         Determines how large the object's hit counter must be in order to be considered as initialized, and get returned to the user as a real object.
         It must be smaller than `hit_counter_max` or otherwise the object would never be initialized.

         If set to 0, objects will get returned to the user as soon as they are detected for the first time,
         which can be problematic as this can result in objects appearing and immediately dissapearing.

         Defaults to `hit_counter_max / 2`
    pointwise_hit_counter_max : int, optional
        Each tracked object keeps track of how often the points it's tracking have been getting matched.
        Points that are getting matched (`pointwise_hit_counter > 0`) are said to be live, and points which aren't (`pointwise_hit_counter = 0`)
        are said to not be live.

        This is used to determine things like which individual points in a tracked object get drawn by [`draw_tracked_objects`][norfair.drawing.draw_tracked_objects] and which don't.
        This argument defines how large the inertia for each point of a tracker can grow.
    detection_threshold : float, optional
        Sets the threshold at which the scores of the points in a detection being fed into the tracker must dip below to be ignored by the tracker.
    filter_factory : FilterFactory, optional
        This parameter can be used to change what filter the [`TrackedObject`][norfair.tracker.TrackedObject] instances created by the tracker will use.
        Defaults to [`OptimizedKalmanFilterFactory()`][norfair.filter.OptimizedKalmanFilterFactory]
    past_detections_length : int, optional
        How many past detections to save for each tracked object.
        Norfair tries to distribute these past detections uniformly through the object's lifetime so they're more representative.
        Very useful if you want to add metric learning to your model, as you can associate an embedding to each detection and access them in your distance function.
    reid_distance_function: Optional[Callable[["TrackedObject", "TrackedObject"], float]]
        Function used by the tracker to determine the ReID distance between newly detected trackers and unmatched trackers by the distance function.

        This function should take 2 input arguments, the first being tracked objects in the initialization phase of type [`TrackedObject`][norfair.tracker.TrackedObject],
        and the second being tracked objects that have been unmatched of type [`TrackedObject`][norfair.tracker.TrackedObject]. It returns a `float` with the distance it
        calculates.
    reid_distance_threshold: float
        Defines what is the maximum ReID distance that can constitute a match.

        Tracked objects whose distance is above this threshold won't be merged, if they are the oldest tracked object will be maintained
        with the position of the new tracked object.
    reid_hit_counter_max: Optional[int]
        Each tracked object keeps an internal ReID hit counter which tracks how often it's getting recognized by another tracker,
        each time it gets a match this counter goes up, and each time it doesn't it goes down. If it goes below 0 the object gets destroyed.
        If used, this argument (`reid_hit_counter_max`) defines how long an object can live without getting matched to any detections, before it is destroyed.
    """

    def __init__(
        self,
        distance_function: Union[str, Callable[["Detection", "TrackedObject"], float]],
        distance_threshold: float,
        hit_counter_max: int = 15,
        initialization_delay: Optional[int] = None,
        pointwise_hit_counter_max: int = 4,
        detection_threshold: float = 0,
        filter_factory: FilterFactory = OptimizedKalmanFilterFactory(),
        past_detections_length: int = 4,
        reid_distance_function: Optional[
            Callable[["TrackedObject", "TrackedObject"], float]
        ] = None,
        reid_distance_threshold: float = 0,
        reid_hit_counter_max: Optional[int] = None,
    ):
        self.tracked_objects: Sequence["TrackedObject"] = []

        if isinstance(distance_function, str):
            distance_function = get_distance_by_name(distance_function)
        elif isinstance(distance_function, Callable):
            warning(
                "You are using a scalar distance function. If you want to speed up the"
                " tracking process please consider using a vectorized distance"
                f" function such as {AVAILABLE_VECTORIZED_DISTANCES}."
            )
            distance_function = ScalarDistance(distance_function)
        else:
            raise ValueError(
                "Argument `distance_function` should be a string or function but is"
                f" {type(distance_function)} instead."
            )
        self.distance_function = distance_function

        self.hit_counter_max = hit_counter_max
        self.reid_hit_counter_max = reid_hit_counter_max
        self.pointwise_hit_counter_max = pointwise_hit_counter_max
        self.filter_factory = filter_factory
        if past_detections_length >= 0:
            self.past_detections_length = past_detections_length
        else:
            raise ValueError(
                f"Argument `past_detections_length` is {past_detections_length} and should be larger than 0."
            )

        if initialization_delay is None:
            self.initialization_delay = int(self.hit_counter_max / 2)
        elif initialization_delay < 0 or initialization_delay >= self.hit_counter_max:
            raise ValueError(
                f"Argument 'initialization_delay' for 'Tracker' class should be an int between 0 and (hit_counter_max = {hit_counter_max}). The selected value is {initialization_delay}.\n"
            )
        else:
            self.initialization_delay = initialization_delay

        self.distance_threshold = distance_threshold
        self.detection_threshold = detection_threshold
        if reid_distance_function is not None:
            self.reid_distance_function = ScalarDistance(reid_distance_function)
        else:
            self.reid_distance_function = reid_distance_function
        self.reid_distance_threshold = reid_distance_threshold
        self._obj_factory = _TrackedObjectFactory()

    def update(
        self,
        detections: Optional[List["Detection"]] = None,
        period: int = 1,
        coord_transformations: Optional[CoordinatesTransformation] = None,
    ) -> List["TrackedObject"]:
        """
        Process detections found in each frame.

        The detections can be matched to previous tracked objects or new ones will be created
        according to the configuration of the Tracker.
        The currently alive and initialized tracked objects are returned

        Parameters
        ----------
        detections : Optional[List[Detection]], optional
            A list of [`Detection`][norfair.tracker.Detection] which represent the detections found in the current frame being processed.

            If no detections have been found in the current frame, or the user is purposely skipping frames to improve video processing time,
            this argument should be set to None or ignored, as the update function is needed to advance the state of the Kalman Filters inside the tracker.
        period : int, optional
            The user can chose not to run their detector on all frames, so as to process video faster.
            This parameter sets every how many frames the detector is getting ran,
            so that the tracker is aware of this situation and can handle it properly.

            This argument can be reset on each frame processed,
            which is useful if the user is dynamically changing how many frames the detector is skipping on a video when working in real-time.
        coord_transformations: Optional[CoordinatesTransformation]
            The coordinate transformation calculated by the [MotionEstimator][norfair.camera_motion.MotionEstimator].

        Returns
        -------
        List[TrackedObject]
            The list of active tracked objects.
        """
        pass

    @property
    def current_object_count(self) -> int:
        """Number of active TrackedObjects"""
        pass

    @property
    def total_object_count(self) -> int:
        """Total number of TrackedObjects initialized in the by this Tracker"""
        pass

    def get_active_objects(self) -> List["TrackedObject"]:
        """Get the list of active objects

        Returns
        -------
        List["TrackedObject"]
            The list of active objects
        """
        pass

    def _update_objects_in_place(
        self,
        distance_function,
        distance_threshold,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
        period: int,
    ):
        pass

    def match_dets_and_objs(self, distance_matrix: np.ndarray, distance_threshold):
        """Matches detections with tracked_objects from a distance matrix

        I used to match by minimizing the global distances, but found several
        cases in which this was not optimal. So now I just match by starting
        with the global minimum distance and matching the det-obj corresponding
        to that distance, then taking the second minimum, and so on until we
        reach the distance_threshold.

        This avoids the the algorithm getting cute with us and matching things
        that shouldn't be matching just for the sake of minimizing the global
        distance, which is what used to happen
        """
        pass


class _TrackedObjectFactory:
    global_count = 0

    def __init__(self) -> None:
        self.count = 0
        self.initializing_count = 0

    def create(
        self,
        initial_detection: "Detection",
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory: "FilterFactory",
        past_detections_length: int,
        reid_hit_counter_max: Optional[int],
        coord_transformations: CoordinatesTransformation,
    ) -> "TrackedObject":
        pass

    def get_initializing_id(self) -> int:
        pass

    def get_ids(self) -> Tuple[int, int]:
        pass


class TrackedObject:
    """
    The objects returned by the tracker's `update` function on each iteration.

    They represent the objects currently being tracked by the tracker.

    Users should not instantiate TrackedObjects manually;
    the Tracker will be in charge of creating them.

    Attributes
    ----------
    estimate : np.ndarray
        Where the tracker predicts the point will be in the current frame based on past detections.
        A numpy array with the same shape as the detections being fed to the tracker that produced it.
    id : Optional[int]
        The unique identifier assigned to this object by the tracker. Set to `None` if the object is initializing.
    global_id : Optional[int]
        The globally unique identifier assigned to this object. Set to `None` if the object is initializing
    last_detection : Detection
        The last detection that matched with this tracked object.
        Useful if you are storing embeddings in your detections and want to do metric learning, or for debugging.
    last_distance : Optional[float]
        The distance the tracker had with the last object it matched with.
    age : int
        The age of this object measured in number of frames.
    live_points :
        A boolean mask with shape `(n_points,)`. Points marked as `True` have recently been matched with detections.
        Points marked as `False` haven't and are to be considered stale, and should be ignored.

        Functions like [`draw_tracked_objects`][norfair.drawing.draw_tracked_objects] use this property to determine which points not to draw.
    initializing_id : int
        On top of `id`, objects also have an `initializing_id` which is the id they are given internally by the `Tracker`;
        this id is used solely for debugging.

        Each new object created by the `Tracker` starts as an uninitialized `TrackedObject`,
        which needs to reach a certain match rate to be converted into a full blown `TrackedObject`.
        `initializing_id` is the id temporarily assigned to `TrackedObject` while they are getting initialized.
    """

    def __init__(
        self,
        obj_factory: _TrackedObjectFactory,
        initial_detection: "Detection",
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory: "FilterFactory",
        past_detections_length: int,
        reid_hit_counter_max: Optional[int],
        coord_transformations: Optional[CoordinatesTransformation] = None,
    ):
        if not isinstance(initial_detection, Detection):
            raise ValueError(
                f"\n[red]ERROR[/red]: The detection list fed into `tracker.update()` should be composed of {Detection} objects not {type(initial_detection)}.\n"
            )
        self._obj_factory = obj_factory
        self.dim_points = initial_detection.absolute_points.shape[1]
        self.num_points = initial_detection.absolute_points.shape[0]
        self.hit_counter_max: int = hit_counter_max
        self.pointwise_hit_counter_max: int = max(pointwise_hit_counter_max, period)
        self.initialization_delay = initialization_delay
        self.detection_threshold: float = detection_threshold
        self.initial_period: int = period
        self.hit_counter: int = period
        self.reid_hit_counter_max = reid_hit_counter_max
        self.reid_hit_counter: Optional[int] = None
        self.last_distance: Optional[float] = None
        self.current_min_distance: Optional[float] = None
        self.last_detection: "Detection" = initial_detection
        self.age: int = 0
        self.is_initializing: bool = self.hit_counter <= self.initialization_delay

        self.initializing_id: Optional[int] = self._obj_factory.get_initializing_id()
        self.id: Optional[int] = None
        self.global_id: Optional[int] = None
        if not self.is_initializing:
            self._acquire_ids()

        if initial_detection.scores is None:
            self.detected_at_least_once_points = np.array([True] * self.num_points)
        else:
            self.detected_at_least_once_points = (
                initial_detection.scores > self.detection_threshold
            )
        self.point_hit_counter: np.ndarray = self.detected_at_least_once_points.astype(
            int
        )
        initial_detection.age = self.age
        self.past_detections_length = past_detections_length
        if past_detections_length > 0:
            self.past_detections: Sequence["Detection"] = [initial_detection]
        else:
            self.past_detections: Sequence["Detection"] = []

        # Create Kalman Filter
        self.filter = filter_factory.create_filter(initial_detection.absolute_points)
        self.dim_z = self.dim_points * self.num_points
        self.label = initial_detection.label
        self.abs_to_rel = None
        if coord_transformations is not None:
            self.update_coordinate_transformation(coord_transformations)

    def tracker_step(self):
        pass

    @property
    def hit_counter_is_positive(self):
        pass

    @property
    def reid_hit_counter_is_positive(self):
        pass

    @property
    def estimate_velocity(self) -> np.ndarray:
        """Get the velocity estimate of the object from the Kalman filter. This velocity is in the absolute coordinate system.

        Returns
        -------
        np.ndarray
            An array of shape (self.num_points, self.dim_points) containing the velocity estimate of the object on each axis.
        """
        pass

    @property
    def estimate(self) -> np.ndarray:
        """Get the position estimate of the object from the Kalman filter.

        Returns
        -------
        np.ndarray
            An array of shape (self.num_points, self.dim_points) containing the position estimate of the object on each axis.
        """
        pass

    def get_estimate(self, absolute=False) -> np.ndarray:
        """Get the position estimate of the object from the Kalman filter in an absolute or relative format.

        Parameters
        ----------
        absolute : bool, optional
            If true the coordinates are returned in absolute format, by default False, by default False.

        Returns
        -------
        np.ndarray
            An array of shape (self.num_points, self.dim_points) containing the position estimate of the object on each axis.

        Raises
        ------
        ValueError
            Alert if the coordinates are requested in absolute format but the tracker has no coordinate transformation.
        """
        pass

    @property
    def live_points(self):
        pass

    def hit(self, detection: "Detection", period: int = 1):
        """Update tracked object with a new detection

        Parameters
        ----------
        detection : Detection
            the new detection matched to this tracked object
        period : int, optional
            frames corresponding to the period of time since last update.
        """
        pass

    def __repr__(self):
        if self.last_distance is None:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {}, init_id: {})"
        else:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {:.2f}, init_id: {})"
        return placeholder_text.format(
            self.id,
            self.age,
            self.hit_counter,
            self.last_distance,
            self.initializing_id,
        )

    def _conditionally_add_to_past_detections(self, detection):
        """Adds detections into (and pops detections away) from `past_detections`

        It does so by keeping a fixed amount of past detections saved into each
        TrackedObject, while maintaining them distributed uniformly through the object's
        lifetime.
        """
        pass

    def merge(self, tracked_object):
        """Merge with a not yet initialized TrackedObject instance"""
        pass

    def update_coordinate_transformation(
        self, coordinate_transformation: CoordinatesTransformation
    ):
        pass

    def _acquire_ids(self):
        pass


class Detection:
    """Detections returned by the detector must be converted to a `Detection` object before being used by Norfair.

    Parameters
    ----------
    points : np.ndarray
        Points detected. Must be a rank 2 array with shape `(n_points, n_dimensions)` where n_dimensions is 2 or 3.
    scores : np.ndarray, optional
        An array of length `n_points` which assigns a score to each of the points defined in `points`.

        This is used to inform the tracker of which points to ignore;
        any point with a score below `detection_threshold` will be ignored.

        This useful for cases in which detections don't always have every point present, as is often the case in pose estimators.
    data : Any, optional
        The place to store any extra data which may be useful when calculating the distance function.
        Anything stored here will be available to use inside the distance function.

        This enables the development of more interesting trackers which can do things like assign an appearance embedding to each
        detection to aid in its tracking.
    label : Hashable, optional
        When working with multiple classes the detection's label can be stored to be used as a matching condition when associating
        tracked objects with new detections. Label's type must be hashable for drawing purposes.
    embedding : Any, optional
        The embedding for the reid_distance.
    """

    def __init__(
        self,
        points: np.ndarray,
        scores: np.ndarray = None,
        data: Any = None,
        label: Hashable = None,
        embedding=None,
    ):
        self.points = validate_points(points)
        self.scores = scores
        self.data = data
        self.label = label
        self.absolute_points = self.points.copy()
        self.embedding = embedding
        self.age = None

    def update_coordinate_transformation(
        self, coordinate_transformation: CoordinatesTransformation
    ):
        pass
