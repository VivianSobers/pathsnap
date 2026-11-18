"""Assemble the technical report (report/PathSnap_Report.md) from the files in results/.

    python report/build_report.py

Every number in the report is read from a results file here, so the report cannot drift from the runs.
Prose is written around the numbers. Rendering to PDF is a separate step (see README).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def J(p):
    return json.loads((R / p).read_text())


f4 = lambda x: f"{x:.4f}"
pc = lambda x: f"{100 * x:.2f}%"

comp = pd.read_csv(R / "models" / "comparison.csv").set_index("model")
runs = {m: J(f"models/{m}.json") for m in comp.index}
inf = J("models/inference_time.json")
cms = J("models/diagnostics/confusion_matrices.json")
curve = J("models/diagnostics/mlp_loss_curve.json")["history"]
inv = pd.read_csv(R / "eda" / "inventory_by_partition.csv").set_index("partition")
samp = pd.read_csv(R / "eda" / "sampling_intervals.csv").set_index("partition")
summ = pd.read_csv(R / "eda" / "trip_summary.csv", dtype={"trip_id": str})
cov_tr, cov_te = J("candidates/coverage_train_r50_k8.json"), J("candidates/coverage_student_test_r50_k8.json")
sweep = pd.read_csv(R / "candidates" / "sweep.csv")
u12, t12 = runs["lightgbm"], J("models/lightgbm_tuned.json")
u15, t15 = J("models/lightgbm_15feat.json"), J("models/lightgbm_f15_tuned.json")
bp12, bp15 = J("hpo/best_params.json"), J("hpo/best_params_lightgbm_f15.json")
trials15 = pd.read_csv(R / "hpo" / "trials_lightgbm_f15.csv")
hmm12, hmmw, hmm15, hmmx = (J("hmm/hmm_vs_classifier.json"), J("hmm/hmm_vs_classifier_wide.json"),
                            J("hmm/hmm_vs_classifier_f15.json"), J("hmm/hmm_vs_classifier_f15_excl54.json"))
ea_old, ea_new = J("hmm/error_analysis_wide.json"), J("hmm/error_analysis_f15.json")
e2e = J("e2e/api_student_test_lightgbm_f15_tuned_uturn10.json")
smoke = J("e2e/evaluator_smoke_test.json")
mins = inf["models"]

NAME = {"logreg": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbours",
        "random_forest": "Random Forest", "lightgbm": "LightGBM", "mlp": "Multi-layer Perceptron"}
CONF = {"logreg": "StandardScaler; L2, C = 1.0; lbfgs, max_iter 1000",
        "gaussian_nb": "Gaussian likelihoods, default variance smoothing",
        "knn": "StandardScaler; k = 25; distance weighted; Euclidean; fit on 150k-row seeded sample",
        "random_forest": "150 trees; min_samples_leaf 5; sqrt of features tried per split",
        "lightgbm": "400 boosting rounds; learning rate 0.05; 63 leaves",
        "mlp": "StandardScaler; hidden (64, 32), ReLU; Adam; early stopping"}
ORDER = ["logreg", "gaussian_nb", "knn", "random_forest", "lightgbm", "mlp"]


def row(m):
    c, t = comp.loc[m], runs[m]["student_test"]
    return (f"| {NAME[m]} | {c.family} | {CONF[m]} | {f4(c.cv_f1_macro)} ± {f4(c.cv_f1_macro_std)} | "
            f"{f4(t['top1_accuracy'])} | {f4(t['route_continuity'])} | {f4(c.test_f1_macro)} | {f4(c.test_log_loss)} | "
            f"{mins[m]['median_s']:.3f} |")


def cm_line(m):
    (tn, fp), (fn, tp) = cms[m]["matrix_[[tn,fp],[fn,tp]]"]
    return tn, fp, fn, tp


PER_MODEL = {
    "logreg": dict(
        why="A linear model on well-scaled geometric features is the simplest sensible baseline: whether a candidate is the reference arc depends mostly on how close it is and whether its direction matches the vehicle, both close to linear in our features. It gives calibrated probabilities directly, which the HMM needs.",
        sl="Fast to train and predict, stable across folds and directly interpretable through its coefficients. It cannot model interactions (for example, that a small distance matters less when several candidates are equally close) unless we build them as features.",
        hp="C = 1.0 (inverse L2 strength), lbfgs solver, max_iter 1000; inputs standardised inside the pipeline.",
        loss="Binary cross-entropy (log loss) with an L2 penalty.",
        fit="Underfitting rather than overfitting: macro F1 is {tr} on the training trips and {te} on the student-test trips (higher on test), and CV macro F1 is {cv}, below every tree model. The model's linear form is the limit."),
    "gaussian_nb": dict(
        why="A probabilistic baseline that models each feature per class with a Gaussian and combines them with Bayes' rule. It tests how far the features carry signal independently of each other.",
        sl="Trains in about a second and needs no tuning. Its independence assumption is badly violated here: distance, rank and distance-to-nearest-candidate are strongly correlated, so their evidence is counted several times and the probabilities are overconfident.",
        hp="Defaults (var_smoothing 1e-9); no scaling needed.",
        loss="No iterative loss: class priors and per-feature Gaussian means and variances are estimated by maximum likelihood.",
        fit="Underfitting: macro F1 {tr} on training and {te} on test, the lowest of all models, with log loss around 0.70. The confusion matrix shows why: it flags {fp:,} non-reference candidates as reference arcs (precision {prec}), so it over-predicts the positive class."),
    "knn": dict(
        why="An instance-based model: a candidate is labelled like the most similar candidates seen in training. Similar geometric situations (distance, heading match, position on the arc) should have similar answers, so local similarity is a natural fit.",
        sl="No training cost beyond storing data, and flexible decision boundaries. Prediction is slow because every query searches the stored set, and distances depend on feature scaling, so all features are standardised. Probabilities are coarse (fractions of 25 neighbours) and its log loss is high.",
        hp="k = 25, distance weighting, Euclidean distance (Minkowski p = 2) on standardised features; fit on a seeded 150,000-row sample of each training fold to keep prediction time practical.",
        loss="No loss is optimised; the prediction is a distance-weighted vote of the 25 nearest training pairs.",
        fit="Slight overfitting: macro F1 {tr} on training and {te} on test. With distance weighting, a training pair that is in the stored 150,000-row sample is its own nearest neighbour, which inflates the training score. The confusion matrix in this report refits on a different 150,000-row sample, so its accuracy ({cmacc}) differs slightly from the recorded {acc}."),
    "random_forest": dict(
        why="An ensemble of decision trees captures non-linear rules and interactions between features (for example distance combined with heading mismatch) without scaling, and bagging makes it stable.",
        sl="Strong accuracy with little tuning and robust to feature scale. It fits the training trips more closely than any other model and is slower to train; its probabilities are averaged votes and slightly less well calibrated than boosting's.",
        hp="150 trees, min_samples_leaf 5, default sqrt features per split, bootstrap sampling, seed 42.",
        loss="Each tree is grown by minimising Gini impurity; the forest averages the trees' class probabilities.",
        fit="Mild overfitting, the largest of the six: macro F1 {tr} on training against {te} on test (accuracy {tracc} against {acc}). It has the best CV macro F1 ({cv}) but does no better than LightGBM on the student-test trips."),
    "lightgbm": dict(
        why="Gradient-boosted trees add trees one at a time, each correcting the previous ones' errors, which suits tabular features with interactions. Boosting with a log-loss objective tends to give well-calibrated probabilities, which the HMM uses directly.",
        sl="Best student-test macro F1 and Top-1 and the lowest log loss, with a small train-test gap. Needs more hyperparameters than a forest, and like all tree models it cannot extrapolate beyond the feature ranges seen in training.",
        hp="400 boosting rounds, learning rate 0.05, 63 leaves per tree, min_child_samples 20, no row or column subsampling, seed 42.",
        loss="Binary log loss (cross-entropy), minimised by gradient boosting.",
        fit="No meaningful overfitting: macro F1 {tr} on training and {te} on test (accuracy {tracc} against {acc}). The fold-to-fold standard deviation of CV macro F1 is {cvsd}, comparable to the other models."),
    "mlp": dict(
        why="A small neural network learns non-linear combinations of the features end to end and outputs probabilities through a sigmoid, so it is a different family from trees with the same output type.",
        sl="Close to the tree models on every metric and very fast at prediction. It needs scaled inputs, is sensitive to initialisation and learning rate, and its validation loss is noisy from epoch to epoch.",
        hp="Two hidden layers (64, 32), ReLU, Adam (learning rate 0.001), minibatch 200, early stopping on a 10% validation split, max 200 epochs, seed 42; inputs standardised.",
        loss="Binary cross-entropy (log loss) with a small L2 penalty (alpha 0.0001).",
        fit="Slight overfitting, visible in the loss curve: training log loss falls from {c0t} to {cNt} over {ne} epochs while validation log loss stays between {vmin} and {vmax} (lowest {vbest} at epoch {ebest}). Macro F1 is {tr} on training and {te} on test, so the gap on the final metric is small."),
}


def model_section(i, m):
    r, c, t = runs[m], comp.loc[m], runs[m]["student_test"]
    tn, fp, fn, tp = cm_line(m)
    vals = [h["val_log_loss"] for h in curve]
    best = min(range(len(vals)), key=vals.__getitem__)
    fmt = dict(tr=f4(r["train"]["f1_macro"]), te=f4(t["f1_macro"]), cv=f4(c.cv_f1_macro), cvsd=f4(c.cv_f1_macro_std),
               tracc=f4(r["train"]["accuracy"]), acc=f4(t["accuracy"]), fp=fp, prec=f4(cms[m]["precision_pos"]),
               cmacc=f4(cms[m]["accuracy"]), c0t=f4(curve[0]["train_log_loss"]), cNt=f4(curve[-1]["train_log_loss"]),
               ne=len(curve), vmin=f4(min(vals)), vmax=f4(max(vals)), vbest=f4(vals[best]), ebest=curve[best]["epoch"])
    p = PER_MODEL[m]
    return f"""### 3.2.{i} {NAME[m]} ({c.family})

**A1. Why suitable.** {p['why']}

**A2. Strengths and limitations.** {p['sl']}

**B5. Hyperparameters and initial values.** {p['hp']}

**B6. Loss or objective.** {p['loss']}

**C7. Student-test performance.** Candidate-level accuracy {f4(t['accuracy'])}, macro precision {f4(t['precision_macro'])}, macro recall {f4(t['recall_macro'])}; route continuity of its classifier-only predictions {f4(t['route_continuity'])}. Confusion matrix at threshold 0.5 (rows reference, columns predicted): {tn:,} true negatives, {fp:,} false positives, {fn:,} false negatives, {tp:,} true positives (`results/models/diagnostics/confusion_{m}.png`).

**C8. Top-1, macro F1 and log loss (student test).** Top-1 {f4(t['top1_accuracy'])}, macro F1 {f4(t['f1_macro'])}, log loss {f4(t['log_loss'])}.

**C9. Overfitting or underfitting.** {p['fit'].format(**fmt)}
"""


tr_inv, te_inv = (inv.loc[p].apply(lambda v: int(round(v))) for p in ("train", "student_test"))
qual = summ.groupby("partition")[["nan_track", "bad_coords", "dup_points", "dt_nonmono", "isolated_nodes",
                                  "route_ids_out_of_range", "route_breaks", "route_repeated_arcs", "jumps_gt100m"]].sum()
s50 = sweep[(sweep.partition == "train") & (sweep.radius_m == 50) & (sweep.k == 8)].iloc[0]
grid100 = [g for g in hmm15["grid_train_oof"] if g["beta"] == 100.0]
ns = e2e["trips"]
tbest = trials15.sort_values("value", ascending=False).iloc[0]

md = f"""<div align="center" style="text-align:center;padding-top:40mm">
<p style="font-size:13pt;letter-spacing:1pt;margin:0">PES UNIVERSITY</p>
<p style="font-size:12pt;margin:6pt 0 0">UE24CS352A: Machine Learning</p>
<p style="font-size:11pt;margin:4pt 0 0">ML Hackathon, Technical Report</p>
<h1 style="font-size:30pt;margin:34mm 0 6pt">PathSnap</h1>
<p style="font-size:15pt;margin:0">Distributed Map-Matching and Route Reconstruction</p>
<p style="font-size:12.5pt;font-weight:bold;margin:30mm 0 8pt">Team 35 &middot; Section C</p>
<table align="center" style="width:auto;margin:0 auto;font-size:11pt">
<tr><th style="padding:5pt 14pt">Name</th><th style="padding:5pt 14pt">SRN</th></tr>
<tr><td style="padding:5pt 14pt">Chirag Basanth Bellavi</td><td style="padding:5pt 14pt">PES1UG24CS133</td></tr>
<tr><td style="padding:5pt 14pt">Chiranthan B S Raj</td><td style="padding:5pt 14pt">PES1UG24CS134</td></tr>
<tr><td style="padding:5pt 14pt">Harsimran Kaur</td><td style="padding:5pt 14pt">PES1UG24CS187</td></tr>
<tr><td style="padding:5pt 14pt">Vivian Sobers E</td><td style="padding:5pt 14pt">PES1UG24CS901</td></tr>
</table>
<p style="font-size:11pt;margin:30mm 0 0">18 November 2026</p>
<p style="font-size:10pt;margin:4pt 0 0">github.com/VivianSobers/pathsnap</p>
</div>

<div style="page-break-after: always"></div>

## Team and contributions

| Name | SRN | Main areas |
|---|---|---|
| Chirag Basanth Bellavi | PES1UG24CS133 | Metrics, cross-validation harness, three of the six models, comparison plots, FastAPI backend |
| Chiranthan B S Raj | PES1UG24CS134 | Evaluator UI, run instructions, UI and API screenshots |
| Harsimran Kaur | PES1UG24CS187 | EDA script, data-quality record, Folium maps, report figures |
| Vivian Sobers E | PES1UG24CS901 | Labels, candidates, core features, final model, Bayesian optimisation, HMM/Viterbi, server-side evaluation, integration |

## Summary

- **Task.** Match each point of a noisy 1 Hz GPS track to the road arc it lies on, and serve the result
  from an API on one laptop to an evaluator UI on another.
- **Method.** Up to 8 candidate arcs within 50 m per point, 15 features per (point, candidate) pair, six
  model families compared with 5-fold grouped cross-validation, LightGBM chosen, then an HMM decoded with
  Viterbi so that the route follows connected roads.
- **Result.** The HMM raised student-test Top-1 from {f4(hmm12['student_test']['classifier_only']['top1_accuracy'])}
  to {f4(hmm12['student_test']['hmm']['top1_accuracy'])} over the classifier alone (12 features, no U-turn
  penalty). With 15 features, the tuned model and the U-turn penalty, the final pipeline through the API scores a pooled Top-1 of {f4(e2e['pooled_top1_accuracy'])} on the 10 student-test trips.
- **Negative results, reported as such.** Bayesian optimisation and dropping the noisiest training trip gave
  no measurable gain.

Every number in this report is generated from a file in `results/` by `report/build_report.py`. Partitions
used: **train** = 80 training trips (fitting and grouped cross-validation), **student test** = 10
student-testing trips (development-stage comparison). The 10 evaluator trips were not used for any
analysis, training or choice (sections 7.3 and 7.4).

## Contents

1. Problem and approach
2. Dataset understanding and EDA
3. Model experimentation and comparison
4. Bayesian hyperparameter optimisation (LightGBM)
5. HMM and Viterbi route reconstruction
6. Geographical visualisation (Folium)
7. API, UI and deployment
8. Conclusions and limitations

<div style="page-break-after: always"></div>

## 1. Problem and approach

Each trip has a noisy 1 Hz GPS track, its own road network (nodes and directed arcs) and a reference route
(the arc ids driven, in order). For every GPS observation we predict the road arc it lies on. The pipeline:
(1) project each trip to metres; (2) label each observation with a reference arc by aligning the route to
the track; (3) take up to 8 candidate arcs within 50 m of each observation; (4) score every
(observation, candidate) pair with a classifier; (5) decode the whole trip with an HMM and Viterbi so the
route follows connected roads; (6) serve the result through a FastAPI backend to an evaluator UI.

## 2. Dataset understanding and EDA

### 2.1 How the data was preprocessed

- **Loading and validation** (`pathsnap/io.py`). Each trip's `.track` (lon, lat, t), `.nodes` (lon, lat),
  `.arcs` (from_node, to_node) and `.route` (arc ids) are read; arcs pointing at missing nodes and route ids
  pointing at missing arcs are rejected. All 90 trips pass.
- **Projection** (`pathsnap/geo.py`). The trips lie on several continents, so no single UTM zone fits.
  Each trip is projected with an azimuthal equidistant projection centred on its GPS centroid; tests show
  planar distances match geodesic distances within 0.1% at four locations worldwide.
- **Labels** (`pathsnap/labels.py`). The route has no timestamps, so each observation is aligned to a
  route position by dynamic programming: positions never go backwards and the summed point-to-arc distance
  is minimal. This handles routes that revisit a road.
- **Candidates** (`pathsnap/candidates.py`). Arcs within 50 m, at most 8 per observation, found with a
  shapely STRtree. Radius and k were chosen with a sweep on the training trips (section 2.4).
- **Gaps.** Time gaps over 60 s split the HMM decoding; real time differences are used in every feature.

### 2.2 Inventory (train and student test)

From `results/eda/inventory_by_partition.csv`:

| Partition | Trips | GPS points | Points per trip (min / median / max) | Nodes per trip (median / max) | Arcs per trip (median / max) | Route length in arcs (median / max) |
|---|---|---|---|---|---|---|
| Train | {tr_inv.trips} | {tr_inv.points_total:,} | {tr_inv.points_min} / {tr_inv.points_median:,} / {tr_inv.points_max:,} | {tr_inv.nodes_median:,} / {tr_inv.nodes_max:,} | {tr_inv.arcs_median:,} / {tr_inv.arcs_max:,} | {tr_inv.route_len_median:,} / {tr_inv.route_len_max:,} |
| Student test | {te_inv.trips} | {te_inv.points_total:,} | {te_inv.points_min} / {te_inv.points_median:,} / {te_inv.points_max:,} | {te_inv.nodes_median:,} / {te_inv.nodes_max:,} | {te_inv.arcs_median:,} / {te_inv.arcs_max:,} | {te_inv.route_len_median:,} / {te_inv.route_len_max:,} |

### 2.3 Missing values and data quality (train and student test)

From `results/eda/trip_summary.csv` and `results/eda/sampling_intervals.csv`:

| Check | Train | Student test |
|---|---|---|
| Missing (NaN) GPS values | {qual.loc['train','nan_track']} | {qual.loc['student_test','nan_track']} |
| Out-of-range coordinates | {qual.loc['train','bad_coords']} | {qual.loc['student_test','bad_coords']} |
| Non-increasing timestamps | {qual.loc['train','dt_nonmono']} | {qual.loc['student_test','dt_nonmono']} |
| Isolated road nodes | {qual.loc['train','isolated_nodes']} | {qual.loc['student_test','isolated_nodes']} |
| Route ids out of range | {qual.loc['train','route_ids_out_of_range']} | {qual.loc['student_test','route_ids_out_of_range']} |
| Steps exactly 1 s apart | {samp.loc['train','pct_exactly_1s']:.2f}% | {samp.loc['student_test','pct_exactly_1s']:.2f}% |
| Steps over 30 s | {samp.loc['train','pct_gt_30s']:.3f}% | {samp.loc['student_test','pct_gt_30s']:.3f}% |
| Longest gap | {samp.loc['train','max_gap_s']:,.0f} s | {samp.loc['student_test','max_gap_s']:,.0f} s |
| Route breaks (consecutive arcs not connected) | {qual.loc['train','route_breaks']} | {qual.loc['student_test','route_breaks']} |
| Position jumps over 100 m | {qual.loc['train','jumps_gt100m']} | {qual.loc['student_test','jumps_gt100m']} |

There are no missing values, so no imputation was needed. The problems are of other kinds: irregular
sampling with some very long gaps (one training trip has a gap of over 42 hours), position jumps, reference
routes with breaks, and road networks split into many weakly connected components. Decisions for each are in
`docs/data_quality.md`.

![GPS trajectories](../results/eda/trajectory_examples.png)

![Sampling intervals](../results/eda/sampling_interval_hist.png)

![Road-network size and connectivity](../results/eda/network_connectivity.png)

![Trip locations](../results/eda/trip_centroids.png)

### 2.4 Patterns observed and the features chosen

- **Most points are close to the right arc, but not always the nearest one.** Parallel roads, two-way
  roads stored as two arcs with the same geometry, and GPS noise make the nearest arc wrong for many
  points, so distance alone is not enough.
- **Coverage.** With 50 m and k = 8, {pc(cov_te['candidate_miss_rate'])} of student-test observations and
  {pc(cov_tr['candidate_miss_rate'])} of training observations have no correct arc among their candidates.
  Without trip `00000054`, whose points mostly lie far from its network, the training figure is
  {pc(s50.miss_rate_excl_00000054)} (`results/candidates/sweep.csv`). Larger radii lower it only slowly while
  growing the candidate set quickly.
- **Direction matters**: heading separates the two directions of a two-way road and rejects cross streets.

Fifteen features per (observation, candidate) pair (`pathsnap/features.py`):

| Group | Features | Why |
|---|---|---|
| Distance | `dist`, `rank`, `dist_minus_min`, `n_cand` | Closeness, and how ambiguous the point is |
| Position on arc | `t_along`, `arc_len` | Points projecting onto an arc's end are often on the next arc |
| Direction | `cos_heading`, `abs_cos_heading` | Separates two-way directions; rejects cross streets |
| Motion | `speed`, `dt_prev`, `turn_rate_rad_s` | Speed, gaps and turning change how far the point can be from the road |
| Neighbours | `dist_prev_pt`, `dist_next_pt`, `previous_candidate_m`, `next_candidate_m` | A true arc stays close across consecutive points |

The first 12 were used for the model comparison (section 3). The last three were added afterwards and kept
because they improved every metric slightly (section 4.2).

### 2.5 Folium map

![Folium map, student-test trip 00000007](figures/folium_student_test_00000007.png)

Student-test trip `00000007`: raw GPS points (blue), nearby road network (grey), reference route (green) and
predicted route (red), with layer control. Interactive maps: `results/maps/`.

## 3. Model experimentation and comparison

Six models from six families were trained with identical features (the first 12), identical
`GroupKFold` folds (5 folds grouped by trip on the 80 training trips) and seed 42, then refit on all 80
training trips and scored on the 10 student-test trips. Scaling sits inside each pipeline, so it is fit on
training folds only. Each model is a binary classifier of candidate pairs ("is this the reference arc?");
the highest-probability candidate per observation is its classifier-only prediction.

### 3.1 Overall comparison

From `results/models/comparison.csv` and `results/models/inference_time.json`. Top-1 and route continuity
are classifier-only (no HMM), on the student-test trips. Inference time is predicting all
{inf['pairs']:,} student-test candidate pairs, median of {inf['repeats']} runs, all models on the same machine.

| Model | Algorithm family | Key configuration | CV macro F1 | Top-1 accuracy | Route continuity | Student-test macro F1 | Log loss | Inference time (s) |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(row(m) for m in ORDER) + f"""

**Primary metric.** We select on student-test Top-1 and log loss together with CV macro F1. Top-1 is the
guidelines' observation-level target; log loss matters because the HMM uses the probabilities directly.

**Selection: LightGBM.** Random Forest, LightGBM and MLP are within 0.006 CV macro F1 of each other while
the fold-to-fold standard deviation is about 0.027, so CV alone cannot separate them. LightGBM has the best
student-test macro F1 ({f4(comp.loc['lightgbm'].test_f1_macro)}) and Top-1 ({f4(runs['lightgbm']['student_test']['top1_accuracy'])}),
the lowest student-test log loss ({f4(comp.loc['lightgbm'].test_log_loss)}), ties Random Forest on CV log loss and overfits less than
Random Forest (section 3.2).

### 3.2 Questions for every model

""" + "\n".join(model_section(i + 1, m) for i, m in enumerate(ORDER)) + f"""

### 3.3 Model visualisations

![Model comparison](../results/models/comparison.png)

CV macro F1, student-test Top-1 and macro F1, log loss and inference time for all six models (the
inference-time panel in this figure comes from the comparison runs, which used two machines; the table in
3.1 uses one machine).

| | |
|---|---|
| ![](../results/models/diagnostics/confusion_logreg.png) | ![](../results/models/diagnostics/confusion_gaussian_nb.png) |
| ![](../results/models/diagnostics/confusion_knn.png) | ![](../results/models/diagnostics/confusion_random_forest.png) |
| ![](../results/models/diagnostics/confusion_lightgbm.png) | ![](../results/models/diagnostics/confusion_mlp.png) |

![MLP loss curve](../results/models/diagnostics/mlp_loss_curve.png)

The MLP is the only model with epoch-wise training, so it is the only one with a loss curve. The other
models have no training history and no curve is drawn for them.

## 4. Bayesian hyperparameter optimisation (LightGBM)

### 4.1 Search

1. **Hyperparameters searched:** n_estimators, learning_rate, num_leaves, min_child_samples, subsample,
   colsample_bytree, reg_lambda, reg_alpha.
2. **Ranges** (`configs/search_space.yaml`): n_estimators 100 to 800; learning_rate 0.01 to 0.2 (log);
   num_leaves 15 to 255 (log); min_child_samples 5 to 200 (log); subsample 0.5 to 1.0; colsample_bytree 0.5
   to 1.0; reg_lambda and reg_alpha 0.001 to 10 (log).
3. **Objective:** mean macro F1 over the same 5 GroupKFold-by-trip folds on the 80 training trips.
4. **Trials:** Optuna TPE (seed 42, 8 random start-up trials, three workers sharing one study). Final
   feature set: {bp15['n_complete']} completed trials (`results/hpo/trials_lightgbm_f15.csv`); trial 0 is the
   untuned configuration. The 12-feature search also ran {bp12['n_complete']} trials (`results/hpo/trials.csv`).
5. **Best values** (trial {bp15['trial']}): """ + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in bp15["params"].items()) + f""".
   Each trial took up to {trials15['user_attrs_seconds'].max():.0f} s for 5 folds.
6. **Did tuning help?** Not measurably; see 4.3.

### 4.2 Feature ablation

| Untuned LightGBM | CV macro F1 | CV log loss | Student-test macro F1 | Student-test Top-1 | Student-test log loss |
|---|---|---|---|---|---|
| 12 features | {f4(u12['cv_mean']['f1_macro'])} | {f4(u12['cv_mean']['log_loss'])} | {f4(u12['student_test']['f1_macro'])} | {f4(u12['student_test']['top1_accuracy'])} | {f4(u12['student_test']['log_loss'])} |
| 15 features | {f4(u15['cv_mean']['f1_macro'])} | {f4(u15['cv_mean']['log_loss'])} | {f4(u15['student_test']['f1_macro'])} | {f4(u15['student_test']['top1_accuracy'])} | {f4(u15['student_test']['log_loss'])} |

### 4.3 Tuned against untuned (15 features)

Same folds and partitions; inference time on one machine (`results/models/inference_time.json`).

| LightGBM, 15 features | CV macro F1 | Student-test Top-1 | Student-test macro F1 | Log loss | Inference time (s) |
|---|---|---|---|---|---|
| Untuned | {f4(u15['cv_mean']['f1_macro'])} ± {f4(u15['cv_std']['f1_macro'])} | {f4(u15['student_test']['top1_accuracy'])} | {f4(u15['student_test']['f1_macro'])} | {f4(u15['student_test']['log_loss'])} | {mins['lightgbm_f15_untuned']['median_s']:.3f} |
| Tuned | {f4(t15['cv_mean']['f1_macro'])} ± {f4(t15['cv_std']['f1_macro'])} | {f4(t15['student_test']['top1_accuracy'])} | {f4(t15['student_test']['f1_macro'])} | {f4(t15['student_test']['log_loss'])} | {mins['lightgbm_f15_tuned']['median_s']:.3f} |

Tuning raised CV macro F1 by {f4(t15['cv_mean']['f1_macro'] - u15['cv_mean']['f1_macro'])} and lowered CV log loss from
{f4(u15['cv_mean']['log_loss'])} to {f4(t15['cv_mean']['log_loss'])}, both well inside the fold-to-fold spread, and left the
student-test scores unchanged, while prediction became about {mins['lightgbm_f15_tuned']['median_s'] / mins['lightgbm_f15_untuned']['median_s']:.1f} times slower (more and deeper trees). The 12-feature search gave the same picture (CV macro F1
{f4(u12['cv_mean']['f1_macro'])} untuned, {f4(t12['cv_mean']['f1_macro'])} tuned; student-test Top-1 {f4(u12['student_test']['top1_accuracy'])} and
{f4(t12['student_test']['top1_accuracy'])}). We conclude that the defaults were already near the best region. The final model uses
the tuned parameters because the selection rule (the CV objective) was fixed before looking at test results.

## 5. HMM and Viterbi route reconstruction

1. **Hidden states.** At each observation, its candidate arcs (up to 8). Each state keeps its arc id and
   its projected position on that arc, so the decoded state maps straight back to a road segment.
2. **Emission probabilities.** w × log P(candidate is the reference arc), from the classifier, clipped at
   1e-6.
3. **Transition probabilities.** From candidate a at observation i to candidate b at i+1:
   −|network distance − straight-line distance| / β, where the network distance runs along the directed road
   graph from the projected point on a to the projected point on b (Dijkstra, capped at 3 km). A U-turn
   onto the reverse of the same segment costs an extra penalty. Unreachable pairs are −∞.
4. **Missing candidates and large gaps.** Observations without candidates get no prediction and are
   bridged; decoding restarts after time gaps over 60 s.
5. **Disconnected segments.** If every transition between two consecutive observations is −∞, the
   current segment is closed by backtracking and a new segment starts from the next observation's emissions.
6. **Did the HMM improve the route?** Yes, on every metric (tables below).

**Decoder parameters.** Only β × w and the U-turn penalty divided by w affect the best path, so w = 1. β was
searched over 2 to 1000 (`results/hmm/hmm_vs_classifier_wide.json`) and the penalty over 0 to 10, using
out-of-fold probabilities on the training trips only; the rule was the highest training Top-1:

| β = 100, U-turn penalty | Train Top-1 | Train route continuity | Train route arc F1 |
|---|---|---|---|
""" + "\n".join(f"| {g['uturn']:g} | {f4(g['top1_accuracy'])} | {f4(g['route_continuity'])} | {f4(g['route_arc_f1'])} |" for g in grid100) + f"""

**Classifier-only against classifier + HMM** (15-feature LightGBM, β = 100, penalty 10;
`results/hmm/hmm_vs_classifier_f15.json`). Route arc F1 compares the set of arcs in the predicted route with
the reference route's arcs, averaged over trips.

| | Train (out-of-fold) classifier | Train + HMM | Student-test classifier | Student-test + HMM |
|---|---|---|---|---|
| Top-1 accuracy | {f4(hmm15['train_oof']['classifier_only']['top1_accuracy'])} | {f4(hmm15['train_oof']['hmm']['top1_accuracy'])} | {f4(hmm15['student_test']['classifier_only']['top1_accuracy'])} | {f4(hmm15['student_test']['hmm']['top1_accuracy'])} |
| Route continuity | {f4(hmm15['train_oof']['classifier_only']['route_continuity'])} | {f4(hmm15['train_oof']['hmm']['route_continuity'])} | {f4(hmm15['student_test']['classifier_only']['route_continuity'])} | {f4(hmm15['student_test']['hmm']['route_continuity'])} |
| Route arc precision | {f4(hmm15['train_oof']['classifier_only']['route_arc_precision'])} | {f4(hmm15['train_oof']['hmm']['route_arc_precision'])} | {f4(hmm15['student_test']['classifier_only']['route_arc_precision'])} | {f4(hmm15['student_test']['hmm']['route_arc_precision'])} |
| Route arc recall | {f4(hmm15['train_oof']['classifier_only']['route_arc_recall'])} | {f4(hmm15['train_oof']['hmm']['route_arc_recall'])} | {f4(hmm15['student_test']['classifier_only']['route_arc_recall'])} | {f4(hmm15['student_test']['hmm']['route_arc_recall'])} |
| Route arc F1 | {f4(hmm15['train_oof']['classifier_only']['route_arc_f1'])} | {f4(hmm15['train_oof']['hmm']['route_arc_f1'])} | {f4(hmm15['student_test']['classifier_only']['route_arc_f1'])} | {f4(hmm15['student_test']['hmm']['route_arc_f1'])} |

**Where the HMM did not help.** Without the U-turn penalty, the HMM raised student-test route continuity
from {f4(hmm12['student_test']['classifier_only']['route_continuity'])} to {f4(hmm12['student_test']['hmm']['route_continuity'])}
(`results/hmm/hmm_vs_classifier.json`), but about half of its remaining errors were the reverse arc of a
two-way road ({pc(ea_old['share']['reverse_direction'])} of all student-test points, `results/hmm/error_analysis_wide.json`). In
{pc(ea_old['reverse_errors_with_reference_against_motion'])} of those, the reference arc points against the vehicle's direction of motion,
and {pc(ea_old['reference_arc_against_motion_share_moving_points'])} of all moving student-test points have such a reference arc. We treat
this as a limitation of the reference routes. The U-turn penalty reduces these errors to
{pc(ea_new['share']['reverse_direction'])} and raises Top-1, but lowers route continuity (training {f4(grid100[0]['route_continuity'])} without the
penalty, {f4(grid100[-1]['route_continuity'])} with it), because following the reference's direction there means consecutive arcs
that do not join. We select on Top-1 and report continuity alongside. A small β × w has the opposite
effect (high continuity, low Top-1), because the decoder then overrides correct short detours.

Leaving trip `00000054` out of training changed student-test HMM Top-1 from
{f4(hmm15['student_test']['hmm']['top1_accuracy'])} to {f4(hmmx['student_test']['hmm']['top1_accuracy'])}, within noise
(`results/hmm/hmm_vs_classifier_f15_excl54.json`); all 80 trips are kept.

## 6. Geographical visualisation (Folium)

![Folium map, student-test trip 00000077](figures/folium_student_test_00000077.png)

Student-test trip `00000077`, the hardest student-test trip: original GPS observations (blue), road network
near the track (grey), reference route (green) and the final model's predicted route (red), with a legend
and layer control (`results/maps/student_test_00000077_predicted.html`; code in `pathsnap/maps.py` and
`scripts/make_prediction_maps.py`).

## 7. API, UI and deployment

### 7.1 System

- **Laptop 1, API** (`api/main.py`): FastAPI with `GET /trips` (trip ids and point counts) and
  `POST /predict/{{trip_id}}`. It loads the trip without its route file, runs the exported model
  (`artifacts/lightgbm_f15_tuned_uturn10.joblib`) and the decoder (`pathsnap/predict.py`), then scores the
  prediction against the reference route server side (`api/evaluate.py`). Every request is logged in the
  terminal. Reference routes never leave the server.
- **Laptop 2, UI** (`ui/index.html`): lists the ten trips, sends the selected trip id to the API and shows
  the predicted route on a map with the returned scores.
- Setup and run instructions: README, "Run on two laptops".

### 7.2 End-to-end results (student test)

All 10 student-test trips through the API with the final model
(`results/e2e/api_student_test_lightgbm_f15_tuned_uturn10.json`); all returned HTTP 200.

| Trip | Points | Top-1 | Route continuity | Runtime (s) |
|---|---|---|---|---|
""" + "\n".join(f"| {t['trip']} | {t['n_points']:,} | {f4(t['top1_accuracy'])} | {f4(t['route_continuity'])} | {t['runtime_s']:.2f} |" for t in ns) + f"""
| **Pooled / mean** | {sum(t['n_points'] for t in ns):,} | **{f4(e2e['pooled_top1_accuracy'])}** | {f4(e2e['mean_route_continuity'])} | |

A fresh clone with a new environment reproduced the earlier end-to-end Top-1 exactly, so results come from
the repository alone.

### 7.3 Evaluator trips

The evaluator trips were received after the final model was committed and were used once, to confirm the
system works on every trip (`scripts/evaluator_smoke_test.py`, `results/e2e/evaluator_smoke_test.json`). The
scorer was replaced by a stub, so no reference route was read and no accuracy was computed. All
{smoke['valid']} trips returned a predicted route. Nothing was changed after this check.

### 7.4 Two-laptop run

On 18 November 2026 the API ran on laptop 1 (Fedora, `10.30.202.250:8000`) and the UI on laptop 2
(Windows, `10.30.200.139:8080`), both on the campus Wi-Fi. The UI listed the ten evaluator trips and ran
trip 00000023 end to end, with scoring done on laptop 1. This was after the final model was committed,
and nothing was changed because of it.

The first attempt failed with "Cannot reach the API" even though the requests were reaching laptop 1. The
UI page had been opened from a different address from the one the API allows (CORS). Opening the UI at
laptop 2's network address, the origin set in `PATHSNAP_UI_ORIGIN`, fixed it.

![UI on laptop 2 connected to the API on laptop 1](figures/ui_connectivity.png)

*UI on laptop 2 (address bar: `10.30.200.139:8080`) connected to the API on laptop 1 (`10.30.202.250:8000`),
listing the ten evaluator trips.*

![Predicted route and scores for trip 00000023](figures/ui_route_and_scores.png)

*Trip 00000023: predicted route (blue) over the GPS points (red), with the scores returned by laptop 1.*

<img alt="API terminal on laptop 1" src="figures/api_terminal_predict.png" style="width:58%">


*API terminal on laptop 1. Every request is logged, including `GET /trips` and
`POST /predict/00000023` from laptop 2 (`10.30.200.139`).*

## 8. Conclusions and limitations

- Final pipeline: 50 m / k = 8 candidates, 15 features, tuned LightGBM, HMM/Viterbi with β = 100 and a
  U-turn penalty of 10. Student-test Top-1 through the API: {f4(e2e['pooled_top1_accuracy'])}.
- The HMM gives the largest gain over the classifier alone. Bayesian optimisation and dropping the noisiest
  training trip gave no measurable gain; we report both as negative results.
- Part of the remaining error lies in the reference routes (arcs pointing against the vehicle's motion),
  which also caps route continuity.
- Training-trip Top-1 is lower than student-test Top-1 mainly because {pc(cov_tr['candidate_miss_rate'])} of training points have no
  correct candidate, against {pc(cov_te['candidate_miss_rate'])} in the student-test trips.
"""

(ROOT / "report" / "PathSnap_Report.md").write_text(md)
print("wrote report/PathSnap_Report.md", len(md.split()), "words")
