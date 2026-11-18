import numpy as np
from pyproj import Geod

from pathsnap.geo import to_metric

GEOD = Geod(ellps="WGS84")


def test_projected_distance_matches_geodesic():
    # Two points inside the dataset's area, a few hundred metres apart.
    lon = np.array([8.595203, 8.601000])
    lat = np.array([49.844053, 49.846500])
    x, y = to_metric(lon, lat)
    planar = np.hypot(x[1] - x[0], y[1] - y[0])
    _, _, geodesic = GEOD.inv(lon[0], lat[0], lon[1], lat[1])
    assert abs(planar - geodesic) / geodesic < 1e-3


def test_coordinate_order_is_lon_then_lat():
    # In UTM 32N, moving north raises y by about 1.11 km per 0.01 degree. x shifts only slightly,
    # because grid north at 8.6 E is rotated about 0.3 degrees from true north (central meridian 9 E).
    x, y = to_metric(np.array([8.6, 8.6]), np.array([49.80, 49.81]))
    dx, dy = x[1] - x[0], y[1] - y[0]
    assert 1100 < dy < 1125
    assert abs(dx) < 0.01 * dy
