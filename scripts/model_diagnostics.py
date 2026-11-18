"""Confusion matrices for every model and a training/validation loss curve for the MLP.

    python -m scripts.model_diagnostics

Confusion matrices: each phase 1 model (first 12 features, same settings as scripts/train_models.py)
and the final model (15 features, tuned) are fit on the 80 training trips and evaluated on the 10
student-test trips at threshold 0.5. MLP curve: the MLP is trained epoch by epoch with partial_fit on
four GroupKFold folds of the training trips and scored on the held-out fold after every epoch.
Writes results/models/diagnostics/.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap.config import load_config
from pathsnap.experiment import load_partition
from pathsnap.features import FEATURES
from scripts.train_models import models

OUT = load_config()["paths"]["results"] / "models" / "diagnostics"
NAMES = {"logreg": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "knn": "KNN",
         "random_forest": "Random Forest", "lightgbm": "LightGBM", "mlp": "MLP",
         "final_lightgbm_f15_tuned": "Final: LightGBM, 15 features, tuned"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def fit_one(name):
    cfg = load_config(); seed = cfg["seed"]
    _, X, y, _ = load_partition(cfg, "train")
    _, tX, ty, _ = load_partition(cfg, "student_test")
    if name == "final_lightgbm_f15_tuned":
        model = models(seed, 4)["lightgbm"][0]()
        params = json.loads((cfg["paths"]["results"] / "hpo" / "best_params_lightgbm_f15.json").read_text())["params"]
        model.set_params(**params, subsample_freq=1); sub = None
    else:
        X, tX = X[:, :12], tX[:, :12]  # phase 1 feature set
        model, sub = models(seed, 4)[name]
        if not hasattr(model, "fit"):
            model = model()
    idx = np.arange(len(y))
    if sub and len(idx) > sub:
        idx = np.random.default_rng(seed).choice(idx, sub, replace=False)
    model.fit(X[idx], y[idx])
    pred = (model.predict_proba(tX)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(ty, pred, labels=[0, 1])
    return name, cm.tolist()


def plot_cm(name, cm):
    cm = np.array(cm); share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=160)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    labels = ["Not reference arc", "Reference arc"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%} of row", ha="center", va="center", fontsize=8.5,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], labels, fontsize=8, color=INK2); ax.set_yticks([0, 1], labels, fontsize=8, color=INK2)
    ax.set_xlabel("Predicted (threshold 0.5)", fontsize=8.5, color=INK2); ax.set_ylabel("Reference", fontsize=8.5, color=INK2)
    ax.set_title(f"{NAMES[name]}\nstudent-test candidate pairs", fontsize=9.5, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"confusion_{name}.png"); plt.close(fig)


def mlp_curve(epochs=60):
    cfg = load_config(); seed = cfg["seed"]
    pairs, X, y, _ = load_partition(cfg, "train")
    X = X[:, :12]
    tr, va = next(GroupKFold(cfg["cv"]["n_splits"]).split(X, y, pairs["trip"]))
    sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    m = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=seed)
    rng = np.random.default_rng(seed); hist = []
    for e in range(1, epochs + 1):
        order = rng.permutation(len(tr))
        for b in range(0, len(order), 200):  # minibatch 200, as in MLPClassifier's default
            sel = order[b:b + 200]
            m.partial_fit(Xtr[sel], y[tr][sel], classes=[0, 1])
        hist.append({"epoch": e, "train_log_loss": log_loss(y[tr], m.predict_proba(Xtr)[:, 1], labels=[0, 1]),
                     "val_log_loss": log_loss(y[va], m.predict_proba(Xva)[:, 1], labels=[0, 1])})
        print("mlp epoch", e, round(hist[-1]["train_log_loss"], 4), round(hist[-1]["val_log_loss"], 4), flush=True)
    return {"setup": "hidden (64, 32), adam, minibatch 200, 12 features, GroupKFold fold 0 of 5 as validation",
            "seed": seed, "train_trips": int(pairs["trip"].iloc[tr].nunique()),
            "val_trips": int(pairs["trip"].iloc[va].nunique()), "history": hist}


def plot_curve(c):
    h = c["history"]; e = [r["epoch"] for r in h]
    tl = [r["train_log_loss"] for r in h]; vl = [r["val_log_loss"] for r in h]
    best = int(np.argmin(vl))
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=160)
    ax.plot(e, tl, color=BLUE, lw=2); ax.plot(e, vl, color=ORANGE, lw=2)
    ax.text(e[-1] + 0.6, tl[-1], "Training", color=INK2, fontsize=8, va="center")
    ax.text(e[-1] + 0.6, vl[-1], "Validation", color=INK2, fontsize=8, va="center")
    ax.scatter([e[best]], [vl[best]], s=40, color=ORANGE, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"lowest validation loss\nepoch {e[best]}: {vl[best]:.4f}", (e[best], vl[best]),
                xytext=(-150, -38), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Epoch", fontsize=8.5, color=INK2); ax.set_ylabel("Log loss", fontsize=8.5, color=INK2)
    ax.set_title(f"MLP log loss per epoch\n{c['train_trips']} training trips, {c['val_trips']} validation trips (grouped by trip)",
                 fontsize=9.5, color=INK, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8); ax.set_xlim(0, e[-1] + 8)
    fig.tight_layout(); fig.savefig(OUT / "mlp_loss_curve.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["logreg", "gaussian_nb", "knn", "random_forest", "lightgbm", "mlp", "final_lightgbm_f15_tuned"]
    with ProcessPoolExecutor(4) as ex:
        curve_future = ex.submit(mlp_curve)
        cms = dict(ex.map(fit_one, names))
        curve = curve_future.result()
    summary = {}
    for n, cm in cms.items():
        plot_cm(n, cm)
        (tn, fp), (fn, tp) = cm
        summary[n] = {"matrix_[[tn,fp],[fn,tp]]": cm, "accuracy": (tn + tp) / (tn + fp + fn + tp),
                      "precision_pos": tp / max(tp + fp, 1), "recall_pos": tp / max(tp + fn, 1)}
        print(n, cm, round(summary[n]["accuracy"], 4))
    (OUT / "confusion_matrices.json").write_text(json.dumps(summary, indent=2))
    (OUT / "mlp_loss_curve.json").write_text(json.dumps(curve, indent=2))
    plot_curve(curve)


if __name__ == "__main__":
    main()
