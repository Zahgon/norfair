"""Predefined distances"""
from abc import ABC, abstractmethod
from functools import partial
from logging import warning
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from .tracker import Detection, TrackedObject


class Distance(ABC):
    """
    Abstract class representing a distance.

    Subclasses must implement the method `get_distances`
    """

    @abstractmethod
    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """


class ScalarDistance(Distance):
    """
    ScalarDistance class represents a distance that is calculated pointwise.

    Parameters
    ----------
    distance_function : Union[Callable[["Detection", "TrackedObject"], float], Callable[["TrackedObject", "TrackedObject"], float]]
        Distance function used to determine the pointwise distance between new candidates and objects.
        This function should take 2 input arguments, the first being a `Union[Detection, TrackedObject]`,
        and the second [TrackedObject][norfair.tracker.TrackedObject]. It has to return a `float` with the distance it calculates.
    """

    def __init__(
        self,
        distance_function: Union[
            Callable[["Detection", "TrackedObject"], float],
            Callable[["TrackedObject", "TrackedObject"], float],
        ],
    ):
        self.distance_function = distance_function

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        pass


class VectorizedDistance(Distance):
    """
    VectorizedDistance class represents a distance that is calculated in a vectorized way. This means
    that instead of going through every pair and explicitly calculating its distance, VectorizedDistance
    uses the entire vectors to compare to each other in a single operation.

    Parameters
    ----------
    distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance function used to determine the distances between new candidates and objects.
        This function should take 2 input arguments, the first being a `np.ndarray` and the second
        `np.ndarray`. It has to return a `np.ndarray` with the distance matrix it calculates.
    """

    def __init__(
        self,
        distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.distance_function = distance_function

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        pass

    def _compute_distance(
        self, stacked_candidates: np.ndarray, stacked_objects: np.ndarray
    ) -> np.ndarray:
        """
        Method that computes the pairwise distances between new candidates and objects.
        It is intended to use the entire vectors to compare to each other in a single operation.

        Parameters
        ----------
        stacked_candidates : np.ndarray
            np.ndarray containing a stack of candidates to be compared with the stacked_objects.
        stacked_objects : np.ndarray
            np.ndarray containing a stack of objects to be compared with the stacked_objects.

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        pass


class ScipyDistance(VectorizedDistance):
    """
    ScipyDistance class extends VectorizedDistance for the use of Scipy's vectorized distances.

    This class uses `scipy.spatial.distance.cdist` to calculate distances between two `np.ndarray`.

    Parameters
    ----------
    metric : str, optional
        Defines the specific Scipy metric to use to calculate the pairwise distances between
        new candidates and objects.

    Other kwargs are passed through to cdist

    See Also
    --------
    [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    """

    def __init__(self, metric: str = "euclidean", **kwargs):
        self.metric = metric
        super().__init__(distance_function=partial(cdist, metric=self.metric, **kwargs))


def frobenius(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Frobernius norm on the difference of the points in detection and the estimates in tracked_object.

    The Frobenius distance and norm are given by:

    $$
    d_f(a, b) = ||a - b||_F
    $$

    $$
    ||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    pass


def mean_euclidean(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Average euclidean distance between the points in detection and estimates in tracked_object.

    $$
    d(a, b) = \\frac{\\sum_{i=0}^N ||a_i - b_i||_2}{N}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    pass


def mean_manhattan(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Average manhattan distance between the points in detection and the estimates in tracked_object

    Given by:

    $$
    d(a, b) = \\frac{\\sum_{i=0}^N ||a_i - b_i||_1}{N}
    $$

    Where $||a||_1$ is the manhattan norm.

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        a tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    pass


def _boxes_area(boxes: np.ndarray) -> np.ndarray:
    """
    Calculate the area of bounding boxes.
    """
    pass


def _validate_bboxes(bboxes: np.ndarray):
    """
    Validate that bounding boxes are well formed.
    """
    pass


def iou(candidates: np.ndarray, objects: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of bounding boxes. Both sets of boxes are expected
    to be in `[x_min, y_min, x_max, y_max]` format.

    Normal IoU is 1 when the boxes are the same and 0 when they don't overlap,
    to transform that into a distance that makes sense we return `1 - iou`.

    Parameters
    ----------
    candidates : numpy.ndarray
        (N, 4) numpy.ndarray containing candidates bounding boxes.
    objects : numpy.ndarray
        (K, 4) numpy.ndarray containing objects bounding boxes.

    Returns
    -------
    numpy.ndarray
        (N, K) numpy.ndarray of `1 - iou` between candidates and objects.
    """
    pass


iou_opt = iou  # deprecated


_SCALAR_DISTANCE_FUNCTIONS = {
    "frobenius": frobenius,
    "mean_manhattan": mean_manhattan,
    "mean_euclidean": mean_euclidean,
}
_VECTORIZED_DISTANCE_FUNCTIONS = {
    "iou": iou,
    "iou_opt": iou,  # deprecated
}
_SCIPY_DISTANCE_FUNCTIONS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
AVAILABLE_VECTORIZED_DISTANCES = (
    list(_VECTORIZED_DISTANCE_FUNCTIONS.keys()) + _SCIPY_DISTANCE_FUNCTIONS
)


def get_distance_by_name(name: str) -> Distance:
    """
    Select a distance by name.

    Parameters
    ----------
    name : str
        A string defining the metric to get.

    Returns
    -------
    Distance
        The distance object.
    """
    pass


def create_keypoints_voting_distance(
    keypoint_distance_threshold: float, detection_threshold: float
) -> Callable[["Detection", "TrackedObject"], float]:
    """
    Construct a keypoint voting distance function configured with the thresholds.

    Count how many points in a detection match the with a tracked_object.
    A match is considered when distance between the points is < `keypoint_distance_threshold`
    and the score of the last_detection of the tracked_object is > `detection_threshold`.
    Notice the if multiple points are tracked, the ith point in detection can only match the ith
    point in the tracked object.

    Distance is 1 if no point matches and approximates 0 as more points are matched.

    Parameters
    ----------
    keypoint_distance_threshold: float
        Points closer than this threshold are considered a match.
    detection_threshold: float
        Detections and objects with score lower than this threshold are ignored.

    Returns
    -------
    Callable
        The distance funtion that must be passed to the Tracker.
    """
    pass


def create_normalized_mean_euclidean_distance(
    height: int, width: int
) -> Callable[["Detection", "TrackedObject"], float]:
    """
    Construct a normalized mean euclidean distance function configured with the max height and width.

    The result distance is bound to [0, 1] where 1 indicates oposite corners of the image.

    Parameters
    ----------
    height: int
        Height of the image.
    width: int
        Width of the image.

    Returns
    -------
    Callable
        The distance funtion that must be passed to the Tracker.
    """
    pass


__all__ = [
    "frobenius",
    "mean_manhattan",
    "mean_euclidean",
    "iou",
    "iou_opt",
    "get_distance_by_name",
    "create_keypoints_voting_distance",
    "create_normalized_mean_euclidean_distance",
]
