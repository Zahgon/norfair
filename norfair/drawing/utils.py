from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .drawer import Drawable


def _centroid(tracked_points: np.ndarray) -> Tuple[int, int]:
    pass


def _build_text(drawable: "Drawable", draw_labels, draw_ids, draw_scores):
    pass
