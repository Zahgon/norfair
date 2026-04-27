from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None  # type: ignore[assignment,misc]


class FilterFactory(ABC):
    """Abstract class representing a generic Filter factory

    Subclasses must implement the method `create_filter`
    """

    @abstractmethod
    def create_filter(self, initial_detection: np.ndarray):
        pass


class FilterPyKalmanFilterFactory(FilterFactory):
    """
    This class can be used either to change some parameters of the [KalmanFilter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)
    that the tracker uses, or to fully customize the predictive filter implementation to use (as long as the methods and properties are compatible).

    The former case only requires changing the default parameters upon tracker creation: `tracker = Tracker(..., filter_factory=FilterPyKalmanFilterFactory(R=100))`,
    while the latter requires creating your own class extending `FilterPyKalmanFilterFactory`, and rewriting its `create_filter` method to return your own customized filter.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix, by default 4.0
    Q : float, optional
        Multiplier for the process uncertainty, by default 0.1
    P : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables, by default 10.0

    See Also
    --------
    [`filterpy.KalmanFilter`](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html).
    """

    def __init__(self, R: float = 4.0, Q: float = 0.1, P: float = 10.0):
        self.R = R
        self.Q = Q
        self.P = P

    def create_filter(self, initial_detection: np.ndarray) -> KalmanFilter:
        """
        This method returns a new predictive filter instance with the current setup, to be used by each new [`TrackedObject`][norfair.tracker.TrackedObject] that is created.
        This predictive filter will be used to estimate speed and future positions of the object, to better match the detections during its trajectory.

        Parameters
        ----------
        initial_detection : np.ndarray
            numpy array of shape `(number of points per object, 2)`, corresponding to the [`Detection.points`][norfair.tracker.Detection] of the tracked object being born,
            which shall be used as initial position estimation for it.

        Returns
        -------
        KalmanFilter
            The kalman filter
        """
        pass


class NoFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

    def predict(self):
        pass

    def update(self, detection_points_flatten, R=None, H=None):

        pass


class NoFilterFactory(FilterFactory):
    """
    This class allows the user to try Norfair without any predictive filter or velocity estimation.

    This track only by comparing the position of the previous detections to the ones in the current frame.

    The throughput of this class in FPS is similar to the one achieved by the
    [`OptimizedKalmanFilterFactory`](#optimizedkalmanfilterfactory) class, so this class exists only for
    comparative purposes and it is not advised to use it for tracking on a real application.

    Parameters
    ----------
    FilterFactory : _type_
        _description_
    """

    def create_filter(self, initial_detection: np.ndarray):
        pass


class OptimizedKalmanFilter:
    def __init__(
        self,
        dim_x,
        dim_z,
        pos_variance=10,
        pos_vel_covariance=0,
        vel_variance=1,
        q=0.1,
        r=4,
    ):
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

        # matrix P from Kalman
        self.pos_variance = np.zeros((dim_z, 1)) + pos_variance
        self.pos_vel_covariance = np.zeros((dim_z, 1)) + pos_vel_covariance
        self.vel_variance = np.zeros((dim_z, 1)) + vel_variance

        self.q_Q = q

        self.default_r = r * np.ones((dim_z, 1))

    def predict(self):
        pass

    def update(self, detection_points_flatten, R=None, H=None):

        pass


class OptimizedKalmanFilterFactory(FilterFactory):
    """
    Creates faster Filters than [`FilterPyKalmanFilterFactory`][norfair.filter.FilterPyKalmanFilterFactory].

    It allows the user to create Kalman Filter optimized for tracking and set its parameters.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix.
    Q : float, optional
        Multiplier for the process uncertainty.
    pos_variance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables.
    pos_vel_covariance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to the covariance between position and speed.
    vel_variance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to velocity (not position) variables.
    """

    def __init__(
        self,
        R: float = 4.0,
        Q: float = 0.1,
        pos_variance: float = 10,
        pos_vel_covariance: float = 0,
        vel_variance: float = 1,
    ):
        self.R = R
        self.Q = Q

        # entrances P matrix of KF
        self.pos_variance = pos_variance
        self.pos_vel_covariance = pos_vel_covariance
        self.vel_variance = vel_variance

    def create_filter(self, initial_detection: np.ndarray):
        pass
