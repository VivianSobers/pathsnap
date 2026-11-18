"""Classification, Top-1, candidate miss rate and route continuity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def pair_metrics(y_true, prob) -> dict:
    """Candidate-level binary metrics: is this candidate the reference arc? Threshold 0.5."""
    y_pred = (prob >= 0.5).astype(int)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
    }


def top1_predictions(pairs: pd.DataFrame, prob) -> pd.DataFrame:
    """Highest-scoring candidate arc per (trip, obs). Ties go to the first row, i.e. the nearer arc."""
    df = pairs[["trip", "obs", "arc"]].copy()
    df["prob"] = prob
    idx = df.groupby(["trip", "obs"], sort=False)["prob"].idxmax()
    return df.loc[idx, ["trip", "obs", "arc"]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(top1: pd.DataFrame, ref: dict[str, list[int]]) -> float:
    """Share of all observations whose top-scoring arc is the reference arc.

    Observations with no candidates count as wrong, so the denominator is every observation.
    """
    total = sum(len(v) for v in ref.values())
    correct = 0
    for trip, g in top1.groupby("trip"):
        r = np.asarray(ref[trip])
        correct += int((g["pred_arc"].to_numpy() == r[g["obs"].to_numpy()]).sum())
    return correct / total


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Among consecutive observations whose predicted arc changes, the share where the two arcs connect.

    arcs is the trip's (n_arcs, 2) from/to node table. Connected means to_node(prev) == from_node(next).
    Returns nan when the prediction never changes arc.
    """
    a, b = pred_arcs[:-1], pred_arcs[1:]
    change = a != b
    if not change.any():
        return float("nan")
    return float(np.mean(arcs[a[change], 1] == arcs[b[change], 0]))
