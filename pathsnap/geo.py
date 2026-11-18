"""Projection to metric coordinates and point-to-segment geometry."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

# The trips lie around 8.6 E, 49.8 N, inside UTM zone 32N. Over a city-sized area the projection's
# scale error is well under 0.1%, so planar distances in this CRS can be read as metres.
INPUT_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32632"

_TO_METRIC = Transformer.from_crs(INPUT_CRS, METRIC_CRS, always_xy=True)


def to_metric(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude (in that order) to UTM 32N x/y in metres."""
    x, y = _TO_METRIC.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)
