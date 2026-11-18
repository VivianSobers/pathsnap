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
    "turn_rate_rad_s", "previous_candidate_m", "next_candidate_m",
]


def trip_geometry(trip: Trip):
    tr = trip.track
    org = local_origin(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    px, py = to_metric(tr["lon"].to_numpy(), tr["lat"].to_numpy(), org)
    nx, ny = to_metric(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy(), org)
    a = trip.arcs["from_node"].to_numpy()
    b = trip.arcs["to_node"].to_numpy()
    return np.column_stack([px, py]), np.column_stack([nx[a], ny[a]]), np.column_stack([nx[b], ny[b]])


def build_trip_features(trip: Trip, radius_m: float, max_k: int, *, gap_limit_s: float = 60,
                        geometry=None) -> tuple[pd.DataFrame, dict]:
    points, starts, ends = trip_geometry(trip) if geometry is None else geometry
    px, py = points.T
    ax, ay = starts.T
    bx, by = ends.T
    t = trip.track["t"].to_numpy()
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
    motion = motion_features(np.column_stack([px, py]), t, gap_limit_s=gap_limit_s)
    gaps = np.diff(t, prepend=t[:1])
    segments = np.cumsum((gaps <= 0) | (gaps > gap_limit_s) | ~np.isfinite(gaps))
    projected = np.column_stack([ax[arc] + tt * ux, ay[arc] + tt * uy])
    context = candidate_context(obs, projected, segments[obs])
    min_d = np.full(n, np.inf); np.minimum.at(min_d, obs, d)

    dpv, _ = point_segment(px[ip][obs], py[ip][obs], ax[arc], ay[arc], bx[arc], by[arc])
    dnx, _ = point_segment(px[inx][obs], py[inx][obs], ax[arc], ay[arc], bx[arc], by[arc])

    df = pd.DataFrame({
        "trip": trip.trip_id, "obs": obs, "arc": arc,
        "dist": d, "rank": rank, "dist_minus_min": d - min_d[obs], "n_cand": n_cand[obs],
        "t_along": tt, "arc_len": alen, "cos_heading": cosh, "abs_cos_heading": np.abs(cosh),
        "speed": speed[obs], "dt_prev": dt_prev[obs], "dist_prev_pt": dpv, "dist_next_pt": dnx,
    })
    df["turn_rate_rad_s"] = motion["turn_rate_rad_s"].to_numpy()[obs]
    df = pd.concat([df, context], axis=1)
    info = {"trip": trip.trip_id, "n_obs": n, "obs_with_cand": int((n_cand > 0).sum())}
    if ref is not None:
        df["label"] = (arc == ref[obs]).astype(np.int8)
        hit = np.zeros(n, bool); hit[obs[df["label"].to_numpy() == 1]] = True
        info["ref_in_candidates"] = int(hit.sum())
        info["ref"] = ref.tolist()
    return df, info


def motion_features(points, timestamps, *, gap_limit_s: float = 60.0) -> pd.DataFrame:
    """Compute GPS motion in projected metres and seconds within each trip."""
    points = np.asarray(points, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.ndim != 1 or points.shape != (len(timestamps), 2):
        raise ValueError("expected one timestamp per two-dimensional point")
    if gap_limit_s <= 0:
        raise ValueError("gap limit must be positive")

    steps = np.diff(points, axis=0, prepend=points[:1])
    gaps = np.diff(timestamps, prepend=timestamps[:1])
    distance = np.linalg.norm(steps, axis=1)
    valid = (gaps > 0) & (gaps <= gap_limit_s) & np.isfinite(distance)
    speed = np.divide(distance, gaps, out=np.zeros_like(distance), where=valid)
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    changes = np.diff(headings, prepend=headings[:1])
    heading_valid = valid & (distance > 0)
    previous_valid = np.r_[False, heading_valid[:-1]][:len(gaps)]
    turn_valid = heading_valid & previous_valid
    wrapped = (changes + np.pi) % (2 * np.pi) - np.pi
    turn_rate = np.divide(
        np.abs(wrapped), gaps, out=np.zeros_like(gaps), where=turn_valid
    )
    return pd.DataFrame({
        "gap_s": np.where(np.isfinite(gaps) & (gaps > 0), gaps, 0.0),
        "speed_m_s": speed,
        "turn_rate_rad_s": turn_rate,
    })


def candidate_context(observation_ids, projected, segment_ids) -> pd.DataFrame:
    """Find the nearest projected candidate on each neighbouring observation.

    Inputs describe one trip in projected metres. Segment IDs keep comparisons
    within uninterrupted portions of the track.
    """
    observation_ids = np.asarray(observation_ids)
    projected = np.asarray(projected, dtype=float)
    segment_ids = np.asarray(segment_ids)
    if projected.shape != (len(observation_ids), 2) or segment_ids.shape != observation_ids.shape:
        raise ValueError("expected one projection and segment id per candidate")

    groups = pd.DataFrame({"segment": segment_ids, "observation": observation_ids}).groupby(
        ["segment", "observation"], sort=False
    ).indices
    previous = np.full(len(projected), np.nan)
    following = np.full(len(projected), np.nan)
    for (segment, observation), current in groups.items():
        for offset, result in ((-1, previous), (1, following)):
            neighbour = groups.get((segment, observation + offset))
            if neighbour is not None:
                changes = projected[current, None, :] - projected[neighbour, :]
                result[current] = np.linalg.norm(changes, axis=2).min(axis=1)
    return pd.DataFrame({"previous_candidate_m": previous, "next_candidate_m": following})
