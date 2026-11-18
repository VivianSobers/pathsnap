"""Candidate miss rate and candidate-set size for a grid of radius and k (no model involved).

Uses the 80 training trips and the 10 student-testing trips. Writes results/candidates/sweep.csv.
The choice of radius and k is made on the training trips; student-test rows are reported alongside.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from pathsnap.config import load_config
from pathsnap.features import build_trip_features
from pathsnap.io import list_trips, load_trip

RADII = [20, 30, 50, 75, 100, 150]
KS = [4, 8, 12, 16]


def _one(args):
    part, path, radius, k = args
    df, info = build_trip_features(load_trip(path), radius, k)
    return {"partition": part, "trip": info["trip"], "radius_m": radius, "k": k, "n_obs": info["n_obs"],
            "pairs": len(df), "obs_with_cand": info["obs_with_cand"], "ref_in_cand": info["ref_in_candidates"]}


def main():
    cfg = load_config()
    jobs = [(part, p, r, k) for part in ["train", "student_test"] for p in list_trips(cfg["paths"][part])
            for r, k in itertools.product(RADII, KS)]
    with ProcessPoolExecutor(12) as ex:
        rows = pd.DataFrame(list(ex.map(_one, jobs, chunksize=8)))
    rows.to_csv(cfg["paths"]["results"] / "candidates" / "sweep_by_trip.csv", index=False)
    g = rows.groupby(["partition", "radius_m", "k"]).sum(numeric_only=True).reset_index()
    g["candidate_miss_rate"] = 1 - g["ref_in_cand"] / g["n_obs"]
    g["mean_candidates_per_obs"] = g["pairs"] / g["n_obs"]
    # The same rate without trip 00000054, whose GPS points mostly lie far from its network.
    r2 = rows[rows.trip != "00000054"].groupby(["partition", "radius_m", "k"]).sum(numeric_only=True).reset_index()
    g["miss_rate_excl_00000054"] = 1 - r2["ref_in_cand"] / r2["n_obs"]
    g = g[["partition", "radius_m", "k", "candidate_miss_rate", "miss_rate_excl_00000054", "mean_candidates_per_obs", "pairs"]]
    g.to_csv(cfg["paths"]["results"] / "candidates" / "sweep.csv", index=False)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
