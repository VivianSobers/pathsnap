"""Align GPS observations to reference route arcs to build per-observation labels.

The .route file lists arc ids in driving order but carries no timestamps. We assign every observation
to one position in the route so that positions never go backwards in time and the summed
point-to-arc distance is as small as possible (a monotone alignment solved by dynamic programming).
"""

from __future__ import annotations

import numpy as np

from .geo import point_segment


def align_to_route(px, py, ax, ay, bx, by) -> np.ndarray:
    """Return, for each observation, the index into the route (0..M-1) it is aligned to.

    px, py: observation coordinates (N,). ax..by: route arc endpoints in route order (M,).
    dp[i, j] = cost[i, j] + min(dp[i-1, :j+1]); positions are non-decreasing in i.
    """
    n, m = len(px), len(ax)
    dp = np.empty((n, m), dtype=np.float32)
    prev = None
    for i in range(n):
        d, _ = point_segment(np.full(m, px[i]), np.full(m, py[i]), ax, ay, bx, by)
        d = d.astype(np.float32)
        dp[i] = d if prev is None else d + np.minimum.accumulate(prev)
        prev = dp[i]
    pos = np.empty(n, dtype=np.int64)
    j = int(np.argmin(dp[-1]))
    pos[-1] = j
    for i in range(n - 2, -1, -1):
        j = int(np.argmin(dp[i, : j + 1]))
        pos[i] = j
    return pos
