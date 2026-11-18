"""HMM over candidate arcs and Viterbi decoding.

Hidden state at observation i: which of its candidate arcs the vehicle is on.
Emission: w * log P(candidate is the reference arc), from the classifier.
Transition from candidate a (at obs i) to candidate b (at obs i+1):
    -|network_distance(a -> b) - straight_line_distance(i, i+1)| / beta
where network_distance runs along the road graph from the projected point on a to the projected point
on b (Newson and Krumm style). Unreachable pairs get -inf.
Decoding restarts (a new segment) at time gaps longer than split_s and wherever every transition is
-inf (disconnected roads). Observations without candidates get no prediction and are bridged.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

NEG = -np.inf


def road_graph(nx, ny, arcs):
    """Directed graph over nodes with arc lengths as weights; returns (csr, arc_length)."""
    a, b = arcs[:, 0], arcs[:, 1]
    length = np.hypot(nx[b] - nx[a], ny[b] - ny[a])
    n = len(nx)
    # Duplicate arcs between the same nodes: keep the shortest (csr sums duplicates, so dedupe first).
    key = a.astype(np.int64) * n + b
    order = np.lexsort((length, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    g = csr_matrix((np.maximum(length[sel], 1e-3), (a[sel], b[sel])), shape=(n, n))
    return g, length


def node_distances(g, sources, targets, limit):
    """Shortest-path distance for each (source, target) pair, inf if beyond limit metres."""
    out = np.full(len(sources), np.inf)
    uniq, inv = np.unique(sources, return_inverse=True)
    for s in range(0, len(uniq), 64):
        batch = uniq[s : s + 64]
        D = dijkstra(g, directed=True, indices=batch, limit=limit)
        m = (inv >= s) & (inv < s + len(batch))
        out[m] = D[inv[m] - s, targets[m]]
    return out


def viterbi_segments(steps, emis, trans_fn):
    """Generic Viterbi with restarts.

    steps: list of observation ids (with at least one candidate), in time order.
    emis[k]: log-emission vector for steps[k]. trans_fn(k) -> (n_k, n_{k+1}) log-transition matrix, or
    None to force a restart between k and k+1. Returns the chosen candidate index for each step.
    """
    choice = np.zeros(len(steps), dtype=np.int64)
    seg_start = 0
    score = emis[0].copy()
    back = [None]

    def finish(end):
        j = int(np.argmax(score_hist[end - seg_start]))
        for k in range(end, seg_start - 1, -1):
            choice[k] = j
            if k > seg_start:
                j = back[k - seg_start][j]

    score_hist = [score]
    for k in range(len(steps) - 1):
        T = trans_fn(k)
        if T is not None:
            M = score[:, None] + T
            best_prev = np.argmax(M, axis=0)
            best = M[best_prev, np.arange(M.shape[1])]
        if T is None or not np.isfinite(best).any():
            finish(k)
            seg_start = k + 1
            score = emis[k + 1].copy()
            back = [None]
            score_hist = [score]
            continue
        score = best + emis[k + 1]
        back.append(best_prev)
        score_hist.append(score)
    finish(len(steps) - 1)
    return choice


def decode_trip(obs, arc, t_along, prob, px, py, t, nx, ny, arcs, beta=10.0, w=1.0,
                split_s=60.0, limit_m=3000.0):
    """Decode one trip. obs/arc/t_along/prob: candidate rows sorted by obs. Returns {obs: arc}."""
    g, alen = road_graph(nx, ny, arcs)
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]
    steps = obs[starts]
    emis = [w * np.log(np.clip(prob[s:e], 1e-6, 1.0)) for s, e in zip(starts, ends)]

    # Precompute all network distances needed between consecutive steps in one batch.
    pairs_src, pairs_dst, idx = [], [], []
    for k in range(len(steps) - 1):
        s0, e0, s1, e1 = starts[k], ends[k], starts[k + 1], ends[k + 1]
        A = np.repeat(np.arange(s0, e0), e1 - s1); B = np.tile(np.arange(s1, e1), e0 - s0)
        pairs_src.append(arcs[arc[A], 1]); pairs_dst.append(arcs[arc[B], 0]); idx.append((A, B))
    nd = node_distances(g, np.concatenate(pairs_src), np.concatenate(pairs_dst), limit_m) if idx else []
    offs = np.cumsum([0] + [len(a) for a, _ in idx])

    def trans(k):
        i0, i1 = steps[k], steps[k + 1]
        if t[i1] - t[i0] > split_s:
            return None
        A, B = idx[k]
        ra, rb = arc[A], arc[B]
        net = (1 - t_along[A]) * alen[ra] + nd[offs[k] : offs[k + 1]] + t_along[B] * alen[rb]
        same = (ra == rb) & (t_along[B] >= t_along[A] - 0.05)
        net = np.where(same, np.abs(t_along[B] - t_along[A]) * alen[ra], net)
        gc = np.hypot(px[i1] - px[i0], py[i1] - py[i0])
        lt = np.where(np.isfinite(net), -np.abs(net - gc) / beta, NEG)
        return lt.reshape(ends[k] - starts[k], ends[k + 1] - starts[k + 1])

    ch = viterbi_segments(steps, emis, trans)
    return {int(o): int(arc[starts[k] + c]) for k, (o, c) in enumerate(zip(steps, ch))}
