import os

import numpy as np
from rich import print
from rich.progress import track

from norfair import Detection

try:
    import motmetrics as mm
    import pandas as pd
except ImportError:
    from .utils import DummyMOTMetricsImport

    mm = DummyMOTMetricsImport()
    pandas = DummyMOTMetricsImport()
from collections import OrderedDict


class InformationFile:
    def __init__(self, file_path):
        self.path = file_path
        with open(file_path, "r") as myfile:
            file = myfile.read()
        self.lines = file.splitlines()

    def search(self, variable_name):
        pass


class PredictionsTextFile:
    """Generates a text file with your predicted tracked objects, in the MOTChallenge format.
    It needs the 'input_path', which is the path to the sequence being processed,
    the 'save_path', and optionally the 'information_file' (in case you don't give an
    'information_file', is assumed there is one in the input_path folder).
    """

    def __init__(self, input_path, save_path=".", information_file=None):

        file_name = os.path.split(input_path)[1]

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)

        self.length = information_file.search(variable_name="seqLength")

        predictions_folder = os.path.join(save_path, "predictions")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)

        out_file_name = os.path.join(predictions_folder, file_name + ".txt")
        self.text_file = open(out_file_name, "w+")

        self.frame_number = 1

    def update(self, predictions, frame_number=None):
        pass


class DetectionFileParser:
    """Get Norfair detections from MOTChallenge text files containing detections"""

    def __init__(self, input_path, information_file=None):
        self.frame_number = 1

        # Get detecions matrix data with rows corresponding to:
        # frame, id, bb_left, bb_top, bb_right, bb_down, conf, x, y, z
        detections_path = os.path.join(input_path, "det/det.txt")

        self.matrix_detections = np.loadtxt(detections_path, dtype="f", delimiter=",")
        row_order = np.argsort(self.matrix_detections[:, 0])
        self.matrix_detections = self.matrix_detections[row_order]
        # Coordinates refer to box corners
        self.matrix_detections[:, 4] = (
            self.matrix_detections[:, 2] + self.matrix_detections[:, 4]
        )
        self.matrix_detections[:, 5] = (
            self.matrix_detections[:, 3] + self.matrix_detections[:, 5]
        )

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)
        self.length = information_file.search(variable_name="seqLength")

        self.sorted_by_frame = []
        for frame_number in range(1, self.length + 1):
            self.sorted_by_frame.append(self.get_dets_from_frame(frame_number))

    def get_dets_from_frame(self, frame_number):
        """this function returns a list of norfair Detections class, corresponding to frame=frame_number"""
        pass

    def __iter__(self):
        self.frame_number = 1
        return self

    def __next__(self):
        if self.frame_number <= self.length:
            self.frame_number += 1
            # Frame_number is always 1 unit bigger than the corresponding index in self.sorted_by_frame, and
            # also we just incremented the frame_number, so now is 2 units bigger than the corresponding index
            return self.sorted_by_frame[self.frame_number - 2]

        raise StopIteration()


class Accumulators:
    def __init__(self):
        self.matrixes_predictions = []
        self.paths = []

    def create_accumulator(self, input_path, information_file=None):
        # Check that motmetrics is installed here, so we don't have to process
        # the whole dataset before failing out if we don't.
        pass

    def update(self, predictions=None):
        # Get the tracked boxes from this frame in an array
        pass

    def compute_metrics(
        self,
        metrics=None,
        generate_overall=True,
    ):
        pass

    def save_metrics(self, save_path=".", file_name="metrics.txt"):
        pass

    def print_metrics(self):
        pass


def load_motchallenge(matrix_data, min_confidence=-1):
    """Load MOT challenge data.

    This is a modification of the function load_motchallenge from the py-motmetrics library, defined in io.py
    In this version, the pandas dataframe is generated from a numpy array (matrix_data) instead of a text file.

    Params
    ------
    matrix_data : array  of float that has [frame, id, X, Y, width, height, conf, cassId, visibility] in each row, for each prediction on a particular video

    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """
    pass


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    pass


def eval_motChallenge(matrixes_predictions, paths, metrics=None, generate_overall=True):
    pass
