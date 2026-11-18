"""Grouped cross-validation and experiment logging (config, seed, times, scores)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .features import FEATURES
from .io import list_trips, load_trip
from .metrics import pair_metrics, route_continuity, top1_accuracy, top1_predictions


def load_partition(cfg, part: str):
    pairs = pd.read_parquet(cfg["paths"]["interim"] / f"features_{part}.parquet")
    ref = json.loads((cfg["paths"]["interim"] / f"obs_{part}.json").read_text())
    X = np.nan_to_num(pairs[FEATURES].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return pairs, X, pairs["label"].to_numpy(), ref


def mean_continuity(top1: pd.DataFrame, arcs_by_trip: dict) -> float:
    vals = [route_continuity(g.sort_values("obs")["pred_arc"].to_numpy(), arcs_by_trip[t])
            for t, g in top1.groupby("trip")]
    return float(np.nanmean(vals))


def arcs_by_trip(cfg, part: str) -> dict:
    return {p.name: load_trip(p).arcs.to_numpy() for p in list_trips(cfg["paths"][part])}


def evaluate(model, pairs, X, y, ref, arcs) -> dict:
    t0 = time.perf_counter()
    prob = model.predict_proba(X)[:, 1]
    infer = time.perf_counter() - t0
    top = top1_predictions(pairs, prob)
    m = pair_metrics(y, prob)
    m.update({"top1_accuracy": top1_accuracy(top, ref), "route_continuity": mean_continuity(top, arcs),
              "inference_s": infer, "inference_ms_per_1k_pairs": 1000 * infer / len(X) * 1000})
    return m


def run(name: str, model, cfg, out_dir: Path, fit_subsample: int | None = None) -> dict:
    seed = cfg["seed"]
    pairs, X, y, ref = load_partition(cfg, "train")
    tpairs, tX, ty, tref = load_partition(cfg, "student_test")
    arcs_tr, arcs_te = arcs_by_trip(cfg, "train"), arcs_by_trip(cfg, "student_test")
    rng = np.random.default_rng(seed)

    def fit(m, idx):
        if fit_subsample and len(idx) > fit_subsample:
            idx = rng.choice(idx, fit_subsample, replace=False)
        t0 = time.perf_counter(); m.fit(X[idx], y[idx]); return time.perf_counter() - t0

    folds = []
    for k, (tr, va) in enumerate(GroupKFold(cfg["cv"]["n_splits"]).split(X, y, pairs["trip"])):
        m = clone(model)
        ft = fit(m, tr)
        vtrips = set(pairs["trip"].iloc[va])
        r = evaluate(m, pairs.iloc[va], X[va], y[va], {t: ref[t] for t in vtrips}, arcs_tr)
        r.update({"fold": k, "fit_s": ft, "val_trips": sorted(vtrips)})
        folds.append(r)
        print(name, "fold", k, {kk: round(v, 4) for kk, v in r.items() if isinstance(v, float)}, flush=True)

    final = clone(model)
    ft = fit(final, np.arange(len(X)))
    train_m = evaluate(final, pairs, X, y, ref, arcs_tr)
    test_m = evaluate(final, tpairs, tX, ty, tref, arcs_te)
    keys = [k for k in folds[0] if isinstance(folds[0][k], float)]
    record = {
        "model": name, "seed": seed, "params": {k: repr(v) for k, v in model.get_params().items()},
        "features": FEATURES, "fit_subsample": fit_subsample,
        "candidates": cfg["candidates"], "cv": {"scheme": "GroupKFold by trip", **cfg["cv"]},
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "host": platform.node(),
        "cv_mean": {k: float(np.mean([f[k] for f in folds])) for k in keys},
        "cv_std": {k: float(np.std([f[k] for f in folds])) for k in keys},
        "folds": folds, "final_fit_s": ft, "train": train_m, "student_test": test_m,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record
