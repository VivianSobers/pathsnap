"""Projection to metric coordinates and point-to-segment geometry."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer

# Trips are spread across the world (Europe, the Americas, Asia), so one UTM zone does not fit them all.
# Each trip is projected with an azimuthal equidistant projection centred on its own GPS centroid:
# over a city-sized trip the scale error stays well under 0.1%, so planar distances read as metres.
INPUT_CRS = "EPSG:4326"


@lru_cache(maxsize=256)
def _transformer(lon0: float, lat0: float) -> Transformer:
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs(INPUT_CRS, crs, always_xy=True)


def local_origin(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    """Projection centre for a trip: the mean GPS position, rounded so the transformer can be cached."""
    return round(float(np.mean(lon)), 4), round(float(np.mean(lat)), 4)


def to_metric(lon, lat, origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude (in that order) to x/y metres around ``origin`` (lon0, lat0)."""
    x, y = _transformer(*origin).transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)

def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments, plus the clipped position t in [0, 1] along each segment.

    All arguments are equal-length arrays in metres; row i pairs point i with segment i.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t
