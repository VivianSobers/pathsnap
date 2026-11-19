"""Fit a candidate classifier on training trips and export it for evaluation."""

import argparse
from pathlib import Path

from joblib import dump
import numpy as np

from pathsnap.config import load_config
from pathsnap.experiment import load_partition
from pathsnap.features import FEATURES
from scripts.train_models import models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=models(0, 1), default="lightgbm")
    parser.add_argument("--jobs", type=int, default=2)
    # beta x emission weight = 100 was best on training out-of-fold predictions (results/hmm/).
    parser.add_argument("--beta-m", type=float, default=100)
    parser.add_argument("--uturn-penalty", type=float, default=10.0,
                        help="decoder U-turn penalty; 10 was best on training out-of-fold predictions")
    parser.add_argument("--emission-weight", type=float, default=1.0, help="decoder emission weight w")
    parser.add_argument("--back-tol-m", type=float,
                        help="same-arc backward tolerance in metres; omit for the original 5%% of arc length")
    parser.add_argument("--params-json", type=Path,
                        help="LightGBM parameters from an Optuna best_params*.json file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.beta_m <= 0 or args.jobs < 1 or args.emission_weight <= 0:
        parser.error("--beta-m, --emission-weight and --jobs must be positive")
    config = load_config()
    model, subsample = models(config["seed"], args.jobs)[args.model]
    if not hasattr(model, "fit"):
        model = model()
    if args.params_json:
        import json
        model.set_params(**json.loads(args.params_json.read_text())["params"], subsample_freq=1)
    _, inputs, labels, _ = load_partition(config, "train")
    if subsample and len(labels) > subsample:
        idx = np.random.default_rng(config["seed"]).choice(len(labels), subsample, replace=False)
        inputs, labels = inputs[idx], labels[idx]
    model.fit(inputs, labels)
    output = args.output or config["paths"]["artifacts"] / "model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    dump({
        "model": model, "model_name": args.model, "features": FEATURES,
        "candidates": config["candidates"], "split_s": config["gaps"]["split_s"],
        "beta_m": args.beta_m, "uturn_penalty": args.uturn_penalty,
        "emission_weight": args.emission_weight, "back_tol_m": args.back_tol_m,
        "params_json": str(args.params_json) if args.params_json else None,
    }, output, compress=3)
    print(output)


if __name__ == "__main__":
    main()
