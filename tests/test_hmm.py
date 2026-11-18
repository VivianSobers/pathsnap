import numpy as np
import pandas as pd

from pathsnap.hmm import decode_route
from pathsnap.io import Trip


def test_viterbi_prefers_connected_arcs_and_restarts_after_gap():
    trip = Trip(
        "synthetic", pd.DataFrame({"t": [0, 1, 90, 91]}),
        pd.DataFrame(index=range(5)),
        pd.DataFrame({"from_node": [0, 1, 3], "to_node": [1, 2, 4]}),
        None,
    )
    track = np.array([[0, 0], [10, 0], [12, 0], [13, 0]])
    starts = np.array([[0, 0], [10, 0], [10, 2]])
    ends = np.array([[10, 0], [20, 0], [20, 2]])
    candidates = pd.DataFrame({
        "observation_id": [0, 1, 1, 2],
        "arc_id": [0, 1, 2, 2],
        "position_along_arc": [0.0, 0.0, 0.0, 0.2],
    })
    result = decode_route(trip, (track, starts, ends), candidates, [0.8, 0.2, 0.9, 0.9], split_s=60)
    assert result.tolist() == [0, 1, 2, -1]
