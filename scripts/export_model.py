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
    parser.add_argument("--model", choices=models(0, 1), default="random_forest")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--beta-m", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.beta_m <= 0 or args.jobs < 1:
        parser.error("--beta-m and --jobs must be positive")
    config = load_config()
    model, subsample = models(config["seed"], args.jobs)[args.model]
    if not hasattr(model, "fit"):
        model = model()
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
        "beta_m": args.beta_m,
    }, output, compress=3)
    print(output)


if __name__ == "__main__":
    main()
