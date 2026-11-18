"""Classifier-only against classifier + HMM/Viterbi, using the selected model (LightGBM).

1. Out-of-fold probabilities on the 80 training trips (same GroupKFold as the model comparison), and
   student-test probabilities from a model refit on all 80 training trips.
2. Decoder parameters (beta, emission weight) chosen on the training trips' out-of-fold predictions only.
3. Both decoders scored on train (out-of-fold) and student test. Writes results/hmm/.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config
from pathsnap.experiment import load_partition
from pathsnap.geo import local_origin, to_metric
from pathsnap.hmm import decode_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import route_continuity, top1_predictions

_spec = importlib.util.spec_from_file_location("tm", Path(__file__).with_name("train_models.py"))
_tm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tm)

_TRIPS: dict = {}


def _geom(path):
    tr = load_trip(path)
    org = local_origin(tr.track["lon"].to_numpy(), tr.track["lat"].to_numpy())
    px, py = to_metric(tr.track["lon"].to_numpy(), tr.track["lat"].to_numpy(), org)
    nx, ny = to_metric(tr.nodes["lon"].to_numpy(), tr.nodes["lat"].to_numpy(), org)
    return tr.trip_id, dict(px=px, py=py, t=tr.track["t"].to_numpy(), nx=nx, ny=ny,
                            arcs=tr.arcs.to_numpy(), route=tr.route)


def _decode(args):
    tid, g, cand, beta, w, split_s = args
    return tid, decode_trip(cand["obs"], cand["arc"], cand["t_along"], cand["prob"], g["px"], g["py"],
                            g["t"], g["nx"], g["ny"], g["arcs"], beta=beta, w=w, split_s=split_s)


def score(preds: dict, ref: dict, geoms: dict) -> dict:
    """preds[trip] = {obs: arc}. Top-1 over all observations; continuity; route arc-set P/R/F1."""
    tot = cor = 0; cont = []; P = []; R = []
    for tid, r in ref.items():
        r = np.asarray(r); p = preds.get(tid, {})
        o = np.array(sorted(p), dtype=int); a = np.array([p[i] for i in o], dtype=int)
        tot += len(r); cor += int((a == r[o]).sum()) if len(o) else 0
        if len(a) > 1:
            cont.append(route_continuity(a, geoms[tid]["arcs"]))
        ps, rs = set(a.tolist()), set(geoms[tid]["route"].tolist())
        P.append(len(ps & rs) / max(len(ps), 1)); R.append(len(ps & rs) / max(len(rs), 1))
    P, R = np.array(P), np.array(R)
    f1 = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0)
    return {"top1_accuracy": cor / tot, "route_continuity": float(np.nanmean(cont)),
            "route_arc_precision": float(P.mean()), "route_arc_recall": float(R.mean()),
            "route_arc_f1": float(f1.mean())}


def classifier_only(pairs, prob):
    top = top1_predictions(pairs, prob)
    return {t: dict(zip(g["obs"].tolist(), g["pred_arc"].tolist())) for t, g in top.groupby("trip")}


def hmm_preds(pairs, prob, geoms, beta, w, split_s, workers):
    df = pairs[["trip", "obs", "arc", "t_along"]].copy(); df["prob"] = prob
    jobs = []
    for tid, g in df.groupby("trip", sort=False):
        g = g.sort_values("obs", kind="stable")
        cand = {c: g[c].to_numpy() for c in ["obs", "arc", "t_along", "prob"]}
        jobs.append((tid, geoms[tid], cand, beta, w, split_s))
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_decode, jobs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    cfg = load_config(); seed = cfg["seed"]; split_s = cfg["gaps"]["split_s"]
    out = cfg["paths"]["results"] / "hmm"; out.mkdir(parents=True, exist_ok=True)
    pairs, X, y, ref = load_partition(cfg, "train")
    tpairs, tX, ty, tref = load_partition(cfg, "student_test")
    geoms = dict(map(_geom, list_trips(cfg["paths"]["train"]) + list_trips(cfg["paths"]["student_test"])))

    model = _tm.models(seed, a.workers)["lightgbm"][0]()
    oof = np.zeros(len(X))
    for tr, va in GroupKFold(cfg["cv"]["n_splits"]).split(X, y, pairs["trip"]):
        m = clone(model).fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    final = clone(model).fit(X, y); tprob = final.predict_proba(tX)[:, 1]

    grid = []
    for beta, w in itertools.product([2.0, 5.0, 10.0, 20.0, 50.0], [0.5, 1.0, 2.0]):
        t0 = time.time()
        s = score(hmm_preds(pairs, oof, geoms, beta, w, split_s, a.workers), ref, geoms)
        grid.append({"beta": beta, "w": w, **s, "seconds": time.time() - t0}); print(grid[-1], flush=True)
    best = max(grid, key=lambda r: r["top1_accuracy"])

    t0 = time.time(); test_hmm = hmm_preds(tpairs, tprob, geoms, best["beta"], best["w"], split_s, a.workers)
    hmm_s = time.time() - t0
    result = {
        "model": "lightgbm (untuned, same params as results/models/lightgbm.json)", "seed": seed,
        "split_s": split_s, "selection": "best top1 on training out-of-fold predictions",
        "best": {"beta": best["beta"], "w": best["w"]}, "grid_train_oof": grid,
        "train_oof": {"classifier_only": score(classifier_only(pairs, oof), ref, geoms),
                      "hmm": {k: v for k, v in best.items() if k not in ("beta", "w", "seconds")}},
        "student_test": {"classifier_only": score(classifier_only(tpairs, tprob), tref, geoms),
                         "hmm": score(test_hmm, tref, geoms), "hmm_decode_seconds": hmm_s},
    }
    (out / "hmm_vs_classifier.json").write_text(json.dumps(result, indent=2))
    (out / "student_test_predictions.json").write_text(json.dumps({t: {str(k): v for k, v in p.items()} for t, p in test_hmm.items()}))
    print(json.dumps({k: result[k] for k in ["best", "train_oof", "student_test"]}, indent=2))


if __name__ == "__main__":
    main()
