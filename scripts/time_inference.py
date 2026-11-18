"""Inference time of every model on one machine, so the numbers are comparable.

    python -m scripts.time_inference

Fits each model on the 80 training trips (same settings as the comparison; final model with tuned
parameters on 15 features) and times predict_proba on all student-test candidate pairs, median of 5
runs, single process with 4 threads where the model supports it. Writes results/models/inference_time.json.
"""

import json
import platform
import time

import numpy as np

from pathsnap.config import load_config
from pathsnap.experiment import load_partition
from scripts.train_models import models


def main():
    cfg = load_config(); seed = cfg["seed"]
    _, X, y, _ = load_partition(cfg, "train")
    _, tX, _, _ = load_partition(cfg, "student_test")
    out = {"host": platform.node(), "pairs": int(len(tX)), "repeats": 5, "models": {}}
    for name in ["logreg", "gaussian_nb", "knn", "random_forest", "lightgbm", "mlp", "lightgbm_f15_tuned", "lightgbm_tuned_f12"]:
        if name.startswith("lightgbm_"):
            m = models(seed, 4)["lightgbm"][0](); sub = None
            pfile = "best_params_lightgbm_f15.json" if name == "lightgbm_f15_tuned" else "best_params.json"
            m.set_params(**json.loads((cfg["paths"]["results"] / "hpo" / pfile).read_text())["params"], subsample_freq=1)
        else:
            m, sub = models(seed, 4)[name]
            if not hasattr(m, "fit"):
                m = m()
        cols = slice(None) if name == "lightgbm_f15_tuned" else slice(0, 12)
        idx = np.arange(len(y))
        if sub and len(idx) > sub:
            idx = np.random.default_rng(seed).choice(idx, sub, replace=False)
        m.fit(X[idx][:, cols], y[idx])
        times = []
        for _ in range(5):
            t0 = time.perf_counter(); m.predict_proba(tX[:, cols]); times.append(time.perf_counter() - t0)
        out["models"][name] = {"median_s": float(np.median(times)), "times_s": times}
        print(name, round(float(np.median(times)), 4), flush=True)
    (cfg["paths"]["results"] / "models" / "inference_time.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
