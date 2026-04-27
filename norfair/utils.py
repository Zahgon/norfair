import os
from functools import lru_cache
from logging import warn
from typing import Sequence, Tuple

import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table


def validate_points(points: np.ndarray) -> np.array:
    # If the user is tracking only a single point, reformat it slightly.
    pass


def raise_detection_error_message(points):
    pass


def print_objects_as_table(tracked_objects: Sequence):
    """Used for helping in debugging"""
    pass


def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    pass


def get_cutout(points, image):
    """Returns a rectangular cut-out from a set of points on an image"""
    pass


class DummyOpenCVImport:
    def __getattribute__(self, name):
        raise ImportError(
            r"""[bold red]Missing dependency:[/bold red] You are trying to use Norfair's video features. However, OpenCV is not installed.

Please, make sure there is an existing installation of OpenCV or install Norfair with `pip install norfair\[video]`."""
        )


class DummyMOTMetricsImport:
    def __getattribute__(self, name):
        raise ImportError(
            r"""[bold red]Missing dependency:[/bold red] You are trying to use Norfair's metrics features without the required dependencies.

Please, install Norfair with `pip install norfair\[metrics]`, or `pip install norfair\[metrics,video]` if you also want video features."""
        )


# lru_cache will prevent re-run the function if the message is the same
@lru_cache(maxsize=None)
def warn_once(message):
    """
    Write a warning message only once.
    """
    pass
