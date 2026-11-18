"""Build candidate feature tables for the training and student-testing partitions.

Writes data/interim/features_<partition>.parquet, data/interim/obs_<partition>.json (reference arc per
observation, for observation-level metrics) and results/candidates/coverage_<partition>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from pathsnap.config import load_config
from pathsnap.features import build_trip_features
from pathsnap.io import list_trips, load_trip


def _one(args):
    path, radius, k = args
    return build_trip_features(load_trip(path), radius, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    radius = args.radius or cfg["candidates"]["radius_m"]
    k = args.k or cfg["candidates"]["max_k"]
    out = cfg["paths"]["interim"]; out.mkdir(parents=True, exist_ok=True)
    res = cfg["paths"]["results"] / "candidates"; res.mkdir(parents=True, exist_ok=True)

    for part in ["train", "student_test"]:
        t0 = time.time()
        jobs = [(p, radius, k) for p in list_trips(cfg["paths"][part])]
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(_one, jobs))
        df = pd.concat([r[0] for r in results], ignore_index=True)
        infos = [r[1] for r in results]
        df.to_parquet(out / f"features_{part}.parquet", index=False)
        (out / f"obs_{part}.json").write_text(json.dumps({i["trip"]: i["ref"] for i in infos}))
        n_obs = sum(i["n_obs"] for i in infos)
        summary = {
            "partition": part, "radius_m": radius, "max_k": k, "trips": len(infos), "observations": n_obs,
            "pairs": len(df), "positive_pairs": int(df["label"].sum()),
            "obs_without_candidates": n_obs - sum(i["obs_with_cand"] for i in infos),
            "candidate_miss_rate": 1 - sum(i["ref_in_candidates"] for i in infos) / n_obs,
            "build_seconds": round(time.time() - t0, 1),
        }
        (res / f"coverage_{part}_r{int(radius)}_k{k}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
