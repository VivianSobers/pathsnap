"""Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE).

Objective: mean macro F1 over the same GroupKFold-by-trip folds as the model comparison, on the 80
training trips only. Several worker processes can share one study through a journal file:

    python scripts/tune.py search --worker 0 --jobs 4   (start as many workers as you like)
    python scripts/tune.py finalize                      (after the search)

finalize writes results/hpo/trials.csv and best_params.json, then evaluates the tuned model with the
same harness as the untuned one (results/models/lightgbm_tuned.json).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.model_selection import GroupKFold

from pathsnap.config import REPO_ROOT, load_config
from pathsnap.experiment import load_partition, run
from pathsnap.metrics import pair_metrics

STUDY = "lightgbm_cv_f1"


def _lgbm(**kw):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**kw)


def sample(trial, space):
    out = {}
    for name, s in space.items():
        if s["type"] == "int":
            out[name] = trial.suggest_int(name, s["low"], s["high"], log=s.get("log", False))
        else:
            out[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return out


def storage(cfg):
    d = cfg["paths"]["results"] / "hpo"; d.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(d / "optuna_journal.log")))


def search(cfg, space_cfg, worker, jobs):
    pairs, X, y, _ = load_partition(cfg, "train")
    folds = list(GroupKFold(cfg["cv"]["n_splits"]).split(X, y, pairs["trip"]))
    seed = cfg["seed"]

    def objective(trial):
        p = sample(trial, space_cfg["params"])
        f1s, lls, t0 = [], [], time.time()
        for tr, va in folds:
            m = _lgbm(**p, subsample_freq=1, random_state=seed, n_jobs=jobs, verbose=-1).fit(X[tr], y[tr])
            r = pair_metrics(y[va], m.predict_proba(X[va])[:, 1])
            f1s.append(r["f1_macro"]); lls.append(r["log_loss"])
        trial.set_user_attr("cv_f1_std", float(np.std(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(f1s))

    sampler = optuna.samplers.TPESampler(seed=seed + worker, n_startup_trials=8)
    study = optuna.create_study(study_name=STUDY, storage=storage(cfg), direction="maximize",
                                sampler=sampler, load_if_exists=True)
    if worker == 0 and len(study.trials) == 0:
        # Trial 0 is the untuned configuration, so the search starts from the baseline.
        study.enqueue_trial({"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                             "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                             "reg_lambda": 0.001, "reg_alpha": 0.001})
    study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(
        space_cfg["n_trials"], states=(optuna.trial.TrialState.COMPLETE,))])


def finalize(cfg, jobs):
    study = optuna.load_study(study_name=STUDY, storage=storage(cfg))
    d = cfg["paths"]["results"] / "hpo"
    df = study.trials_dataframe()
    df.to_csv(d / "trials.csv", index=False)
    best = study.best_trial
    (d / "best_params.json").write_text(json.dumps({
        "trial": best.number, "cv_f1_macro": best.value, **best.user_attrs, "params": best.params,
        "n_complete": int((df["state"] == "COMPLETE").sum()), "sampler": "TPE", "seed": cfg["seed"]}, indent=2))
    model = _lgbm(**best.params, subsample_freq=1, random_state=cfg["seed"], n_jobs=jobs, verbose=-1)
    run("lightgbm_tuned", model, cfg, cfg["paths"]["results"] / "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "finalize"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    space_cfg = yaml.safe_load((REPO_ROOT / "configs" / "search_space.yaml").read_text())
    search(cfg, space_cfg, a.worker, a.jobs) if a.mode == "search" else finalize(cfg, a.jobs)


if __name__ == "__main__":
    main()
