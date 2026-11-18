from math import pi

import pandas as pd
import pytest

from pathsnap.features import FEATURES, build_trip_features, candidate_context, motion_features
from pathsnap.io import Trip


def test_motion_features_reset_after_large_gap():
    result = motion_features([[0, 0], [2, 0], [2, 3], [3, 3]], [5, 6, 7, 80])
    assert result["gap_s"].tolist() == [0, 1, 1, 73]
    assert result["speed_m_s"].tolist() == [0, 2, 3, 0]
    assert result["turn_rate_rad_s"].tolist() == pytest.approx([0, 0, pi / 2, 0])


def test_context_uses_nearest_candidate_and_respects_segments():
    result = candidate_context(
        [0, 0, 1, 2], [[1, 0], [4, 0], [2, 0], [2, 0]], [0, 0, 0, 1]
    )
    assert result["previous_candidate_m"].iloc[2] == 1
    assert result["next_candidate_m"].iloc[:2].tolist() == [1, 2]
    assert pd.isna(result["next_candidate_m"].iloc[2])
    assert pd.isna(result["previous_candidate_m"].iloc[3])


def test_candidate_table_exposes_context_for_scoring():
    trip = Trip(
        "sample",
        pd.DataFrame({"lon": [8.6, 8.6001, 8.6002], "lat": [49.8] * 3, "t": [0, 1, 100]}),
        pd.DataFrame({"lon": [8.6, 8.6002], "lat": [49.8, 49.8]}),
        pd.DataFrame({"from_node": [0], "to_node": [1]}),
        None,
    )
    candidates, info = build_trip_features(trip, 50, 8, gap_limit_s=60)
    assert set(FEATURES) <= set(candidates)
    assert info["obs_with_cand"] == 3
    assert candidates["previous_candidate_m"].iloc[1] > 0
    assert pd.isna(candidates["next_candidate_m"].iloc[1])
    assert pd.isna(candidates["previous_candidate_m"].iloc[2])
