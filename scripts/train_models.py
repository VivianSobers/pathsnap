"""Train and evaluate one model family with GroupKFold by trip, then on the student-testing trips.

Usage: python scripts/train_models.py --model lightgbm
Writes results/models/<model>.json with config, seed, per-fold scores, times and student-test scores.
"""

from __future__ import annotations

import argparse

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import load_config
from pathsnap.experiment import run


def models(seed: int, jobs: int):
    # Untuned, sensible defaults. Scaling sits inside the pipeline so it is fit on training folds only.
    return {
        "logreg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)), None),
        "gaussian_nb": (GaussianNB(), None),
        # KNN stores the training set and is slow to query, so it is fit on a seeded 150k-row sample.
        "knn": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=jobs)), 150_000),
        "random_forest": (RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=jobs, random_state=seed), None),
        "lightgbm": (LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=seed, n_jobs=jobs, verbose=-1), None),
        "mlp": (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed)), None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    model, sub = models(cfg["seed"], a.jobs)[a.model]
    run(a.model, model, cfg, cfg["paths"]["results"] / "models", fit_subsample=sub)


if __name__ == "__main__":
    main()
