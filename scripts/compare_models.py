"""Collect results/models/*.json into one comparison table (CSV and Markdown)."""

from __future__ import annotations

import json

import pandas as pd

from pathsnap.config import load_config

FAMILY = {"logreg": "Linear", "gaussian_nb": "Probabilistic", "knn": "Instance-based",
          "random_forest": "Bagging ensemble", "lightgbm": "Boosting", "mlp": "Neural network"}


def main():
    cfg = load_config()
    d = cfg["paths"]["results"] / "models"
    rows = []
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        cm, cs, te = r["cv_mean"], r["cv_std"], r["student_test"]
        rows.append({
            "model": r["model"], "family": FAMILY.get(r["model"], ""),
            "cv_f1_macro": cm["f1_macro"], "cv_f1_macro_std": cs["f1_macro"],
            "cv_top1": cm["top1_accuracy"], "cv_log_loss": cm["log_loss"],
            "test_accuracy": te["accuracy"], "test_precision_macro": te["precision_macro"],
            "test_recall_macro": te["recall_macro"], "test_f1_macro": te["f1_macro"],
            "test_log_loss": te["log_loss"], "test_top1": te["top1_accuracy"],
            "test_route_continuity": te["route_continuity"],
            "train_f1_macro": r["train"]["f1_macro"], "fit_s": r["final_fit_s"],
            "test_inference_s": te["inference_s"], "seed": r["seed"], "commit": r["commit"][:7],
        })
    df = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)
    df.to_csv(d / "comparison.csv", index=False)
    (d / "comparison.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
