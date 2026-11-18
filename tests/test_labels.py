import numpy as np

from pathsnap.geo import point_segment
from pathsnap.labels import align_to_route


def test_point_segment_distance_and_t():
    d, t = point_segment(np.array([5.0, -3.0]), np.array([2.0, 0.0]),
                         np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                         np.array([10.0, 10.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(d, [2.0, 3.0])
    np.testing.assert_allclose(t, [0.5, 0.0])


def test_alignment_is_monotone_even_when_route_revisits_a_road():
    # Route: east along y=0 (arc 0), north (arc 1), then west back along y=0 (arc 2, same road as 0).
    ax = np.array([0.0, 100.0, 100.0]); ay = np.array([0.0, 0.0, 0.0])
    bx = np.array([100.0, 100.0, 0.0]); by = np.array([0.0, 0.0, 0.0])
    # Make arc 1 a short spur so arcs 0 and 2 overlap in space.
    by[1] = 50.0; ay[2] = 0.0
    px = np.array([10.0, 60.0, 100.0, 100.0, 60.0, 10.0])
    py = np.array([1.0, 1.0, 25.0, 25.0, 1.0, 1.0])
    pos = align_to_route(px, py, ax, ay, bx, by)
    assert list(pos) == [0, 0, 1, 1, 2, 2]
