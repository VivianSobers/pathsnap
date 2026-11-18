from math import pi

import pandas as pd
import pytest

from pathsnap.features import candidate_context, motion_features


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
