import numpy as np
import pandas as pd

from pathsnap.metrics import route_continuity, top1_accuracy, top1_predictions


def test_top1_counts_observations_without_candidates_as_wrong():
    pairs = pd.DataFrame({"trip": ["a"] * 4, "obs": [0, 0, 1, 1], "arc": [5, 6, 7, 8]})
    prob = np.array([0.2, 0.9, 0.6, 0.1])
    top = top1_predictions(pairs, prob)
    assert top["pred_arc"].tolist() == [6, 7]
    # Trip has 3 observations; obs 2 had no candidates.
    assert top1_accuracy(top, {"a": [6, 8, 9]}) == 1 / 3


def test_route_continuity():
    arcs = np.array([[0, 1], [1, 2], [5, 6]])
    assert route_continuity(np.array([0, 0, 1, 1]), arcs) == 1.0
    assert route_continuity(np.array([0, 1, 2]), arcs) == 0.5
    assert np.isnan(route_continuity(np.array([2, 2]), arcs))
