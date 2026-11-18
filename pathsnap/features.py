"""Build one feature row per (observation, candidate) pair."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import candidate_pairs
from .geo import local_origin, point_segment, to_metric
from .io import Trip
from .labels import align_to_route

FEATURES = [
    "dist", "rank", "dist_minus_min", "n_cand", "t_along", "arc_len",
    "cos_heading", "abs_cos_heading", "speed", "dt_prev", "dist_prev_pt", "dist_next_pt",
]


def build_trip_features(trip: Trip, radius_m: float, max_k: int) -> tuple[pd.DataFrame, dict]:
    tr = trip.track
    org = local_origin(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    px, py = to_metric(tr["lon"].to_numpy(), tr["lat"].to_numpy(), org)
    t = tr["t"].to_numpy()
    nx, ny = to_metric(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy(), org)
    a = trip.arcs["from_node"].to_numpy(); b = trip.arcs["to_node"].to_numpy()
    ax, ay, bx, by = nx[a], ny[a], nx[b], ny[b]
    n = len(px)

    # Reference arc per observation from the monotone alignment (labels only; never a feature).
    ref = None
    if trip.route is not None:
        r = trip.route
        pos = align_to_route(px, py, ax[r], ay[r], bx[r], by[r])
        ref = r[pos]

    # Observation-level motion: central-difference heading, speed, time since previous fix.
    ip, inx = np.r_[0, np.arange(n - 1)], np.r_[np.arange(1, n), n - 1]
    hx, hy = px[inx] - px[ip], py[inx] - py[ip]
    span = np.hypot(hx, hy)
    dtc = np.maximum(t[inx] - t[ip], 1e-9)
    speed = span / dtc
    dt_prev = np.r_[0.0, np.diff(t)]

    obs, arc, d, tt, rank = candidate_pairs(px, py, ax, ay, bx, by, radius_m, max_k)
    ux, uy = bx[arc] - ax[arc], by[arc] - ay[arc]
    alen = np.hypot(ux, uy)
    denom = np.maximum(alen * span[obs], 1e-9)
    cosh = np.where((span[obs] > 0.5) & (alen > 0), (ux * hx[obs] + uy * hy[obs]) / denom, 0.0)

    n_cand = np.bincount(obs, minlength=n)
    min_d = np.full(n, np.inf); np.minimum.at(min_d, obs, d)

    dpv, _ = point_segment(px[ip][obs], py[ip][obs], ax[arc], ay[arc], bx[arc], by[arc])
    dnx, _ = point_segment(px[inx][obs], py[inx][obs], ax[arc], ay[arc], bx[arc], by[arc])

    df = pd.DataFrame({
        "trip": trip.trip_id, "obs": obs, "arc": arc,
        "dist": d, "rank": rank, "dist_minus_min": d - min_d[obs], "n_cand": n_cand[obs],
        "t_along": tt, "arc_len": alen, "cos_heading": cosh, "abs_cos_heading": np.abs(cosh),
        "speed": speed[obs], "dt_prev": dt_prev[obs], "dist_prev_pt": dpv, "dist_next_pt": dnx,
    })
    info = {"trip": trip.trip_id, "n_obs": n, "obs_with_cand": int((n_cand > 0).sum())}
    if ref is not None:
        df["label"] = (arc == ref[obs]).astype(np.int8)
        hit = np.zeros(n, bool); hit[obs[df["label"].to_numpy() == 1]] = True
        info["ref_in_candidates"] = int(hit.sum())
        info["ref"] = ref.tolist()
    return df, info
