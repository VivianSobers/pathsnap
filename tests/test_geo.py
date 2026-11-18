import numpy as np
import pytest
from pyproj import Geod

from pathsnap.geo import local_origin, to_metric

GEOD = Geod(ellps="WGS84")


@pytest.mark.parametrize("lon0,lat0", [(8.595, 49.844), (-78.17, 0.33), (106.55, 15.40), (18.70, 50.03)])
def test_projected_distance_matches_geodesic_anywhere(lon0, lat0):
    lon = np.array([lon0, lon0 + 0.02]); lat = np.array([lat0, lat0 + 0.015])
    x, y = to_metric(lon, lat, local_origin(lon, lat))
    planar = np.hypot(x[1] - x[0], y[1] - y[0])
    _, _, geodesic = GEOD.inv(lon[0], lat[0], lon[1], lat[1])
    assert np.isfinite(planar)
    assert abs(planar - geodesic) / geodesic < 1e-3


def test_coordinate_order_is_lon_then_lat():
    lon = np.array([8.6, 8.6]); lat = np.array([49.80, 49.81])
    x, y = to_metric(lon, lat, local_origin(lon, lat))
    assert abs(x[1] - x[0]) < 1.0
    assert 1100 < y[1] - y[0] < 1125
