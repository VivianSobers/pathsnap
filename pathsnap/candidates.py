"""Generate candidate arcs near each GPS observation with a spatial index."""

from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree

from .geo import point_segment


def candidate_pairs(px, py, ax, ay, bx, by, radius_m: float, max_k: int):
    """Return (obs_idx, arc_idx, dist, t) for the max_k nearest arcs within radius_m of each observation.

    Arcs are segments a->b in metres. Both directions of a two-way road appear as separate arcs with the
    same geometry; both are kept, and direction is left to the features.
    """
    segs = shapely.linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    tree = STRtree(segs)
    pts = shapely.points(px, py)
    obs, arc = tree.query(pts, predicate="dwithin", distance=radius_m)
    d, t = point_segment(px[obs], py[obs], ax[arc], ay[arc], bx[arc], by[arc])
    keep = d <= radius_m
    obs, arc, d, t = obs[keep], arc[keep], d[keep], t[keep]
    order = np.lexsort((arc, d, obs))  # by observation, then distance, then arc id for ties
    obs, arc, d, t = obs[order], arc[order], d[order], t[order]
    start = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    rank = np.arange(len(obs)) - np.repeat(start, np.diff(np.r_[start, len(obs)]))
    keep = rank < max_k
    return obs[keep], arc[keep], d[keep], t[keep], rank[keep]
