# PathSnap: phase 1 report

Preprocessing, features, technique, the model families we trained, their scores, and which model we take
forward. Every number below is copied from a file in `results/`; the file is named next to each table.

## 1. Preprocessing

**Loading.** `pathsnap/io.py` reads the four files of each trip (`.track`: lon, lat, t; `.nodes`: lon,
lat; `.arcs`: from_node, to_node; `.route`: arc ids in driving order). The loader rejects any arc that
points at a node that does not exist and any route entry that points at an arc that does not exist. All 90
trips (80 train, 10 student test) load and pass these checks. No values are missing and timestamps always
increase.

**Projection.** Longitude and latitude are angles, so distances in degrees are wrong. We first assumed all
trips were in Germany and used UTM zone 32N, but the trips are spread across Europe, the Americas and Asia
(one is in Ecuador at -78 degrees longitude, where the UTM 32N transform returned non-finite values). Each
trip is now projected with an azimuthal equidistant projection centred on its own GPS centroid
(`pathsnap/geo.py`). Tests check that projected distances match geodesic distances to within 0.1% at four
locations around the world.

**Labels.** The `.route` file lists arcs in order but has no timestamps, so we do not know which arc each GPS
point was on. `pathsnap/labels.py` assigns every observation to one position in the route so that positions
never go backwards in time and the total point-to-arc distance is as small as possible (dynamic
programming over the route order). This handles routes that drive the same road twice. These labels are
our reference; the PDFs warn that reference routes may themselves have errors, so they are not perfect
ground truth.

**Candidate generation.** For every observation we take the road arcs within 50 m, at most the 8 nearest,
using a shapely STRtree spatial index (`pathsnap/candidates.py`). Two-way roads appear as two arcs with the
same geometry; we keep both and let the heading features tell the directions apart. Coverage, from
`results/candidates/coverage_<partition>_r50_k8.json`:

| Partition | Observations | Candidate pairs | Observations with no candidate | Candidate miss rate |
|---|---|---|---|---|
| Train | 208,417 | 1,084,832 | 18,230 | 11.92% |
| Student test | 19,160 | 113,639 | 0 | 0.42% |

The train miss rate is driven almost entirely by trip `00000054`, where most points lie more than 50 m
from any arc. We keep the trip (it is real data) and record it as a data-quality issue. Radius and k were
set by hand for this phase; a sweep over both is planned.

## 2. Features and feature selection

One row per (observation, candidate arc) pair, 12 features (`pathsnap/features.py`):

| Group | Features | Why |
|---|---|---|
| Distance | `dist`, `rank`, `dist_minus_min`, `n_cand` | The true arc is usually the nearest, but not always; distance relative to the nearest candidate and the number of competitors say how ambiguous the point is |
| Position on arc | `t_along`, `arc_len` | Points that project onto an arc's end are often on the next arc |
| Direction | `cos_heading`, `abs_cos_heading` | Signed cosine between the vehicle's heading and the arc's direction separates the two directions of a two-way road and rejects cross streets |
| Motion | `speed`, `dt_prev` | Speed and gaps change how far the point can be from the road |
| Neighbours | `dist_prev_pt`, `dist_next_pt` | Distance from the previous and next GPS points to the same arc; a true arc stays close across consecutive points |

The label (`label` = 1 if the candidate is the reference arc) and the trip id are never used as features.
Features were chosen from the geometry of the problem; we have not yet run a data-driven selection step
(feature importance or ablation is planned for the next phase).

## 3. Technique

- **Task.** Binary classification of candidate pairs: is this candidate the reference arc? For each
  observation the candidate with the highest probability is the classifier-only prediction. The HMM and
  Viterbi decoder (next phase) will use these probabilities as emissions.
- **Validation.** `GroupKFold`, 5 folds, grouped by trip, on the 80 training trips, so no trip appears in
  both a training and a validation fold. Each final model is then refit on all 80 training trips and scored
  once on the 10 student-test trips. Evaluator trips are not used at all.
- **Leakage control.** Scaling sits inside each model's pipeline, so it is fit on training folds only.
- **Reproducibility.** Seed 42 for every model; each run writes its parameters, seed, commit, host and
  per-fold scores to `results/models/<model>.json`.
- **Metrics.** Candidate-level accuracy, macro precision, macro recall, macro F1 and log loss; observation-
  level Top-1 accuracy (all observations in the denominator, so those with no candidate count as wrong);
  route continuity (among consecutive points whose predicted arc changes, the share where the two arcs
  connect in the road network).
- **No tuning yet.** All models use untuned default-style settings so the comparison is fair; Bayesian
  optimisation comes next, on the chosen model only.

## 4. Model families

| Model | Family | Configuration |
|---|---|---|
| Logistic Regression | Linear | StandardScaler, C=1.0, max_iter=1000 |
| Gaussian Naive Bayes | Probabilistic | defaults |
| KNN | Instance-based | StandardScaler, k=25, distance-weighted; fit on a seeded 150k-row sample of each training fold |
| Random Forest | Bagging ensemble | 150 trees, min_samples_leaf=5 |
| LightGBM | Boosting | 400 trees, learning rate 0.05, 63 leaves |
| MLP | Neural network | StandardScaler, hidden layers (64, 32), early stopping |

## 5. Results

From `results/models/comparison.csv` (built by `scripts/compare_models.py` from `results/models/<model>.json`). CV = 5-fold GroupKFold by trip on the 80 training trips; Test = the 10 student-test trips, after refitting on all 80. Sorted by CV macro F1.

| Model | Family | CV macro F1 (mean ± sd) | CV Top-1 | CV log loss | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Test log loss | Test Top-1 | Test route continuity | Train macro F1 | Fit time (s) | Test inference (s) | Machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | Bagging ensemble | 0.8984 ± 0.0266 | 0.7603 | 0.1356 | 0.9540 | 0.9238 | 0.9090 | 0.9162 | 0.1153 | 0.8634 | 0.7737 | 0.9551 | 49.8 | 0.242 | Mac mini |
| LightGBM | Boosting | 0.8969 ± 0.0276 | 0.7607 | 0.1355 | 0.9542 | 0.9217 | 0.9128 | 0.9172 | 0.1129 | 0.8651 | 0.7760 | 0.9265 | 14.2 | 0.879 | laptop |
| MLP | Neural network | 0.8930 ± 0.0277 | 0.7574 | 0.1458 | 0.9520 | 0.9170 | 0.9097 | 0.9133 | 0.1187 | 0.8620 | 0.7785 | 0.9116 | 76.2 | 0.012 | Mac mini |
| KNN | Instance-based | 0.8859 ± 0.0272 | 0.7498 | 0.2059 | 0.9471 | 0.9108 | 0.8969 | 0.9037 | 0.1697 | 0.8480 | 0.7590 | 0.9109 | 0.1 | 8.447 | Mac mini |
| Logistic Regression | Linear | 0.8799 ± 0.0289 | 0.7499 | 0.1870 | 0.9412 | 0.8970 | 0.8908 | 0.8939 | 0.1703 | 0.8453 | 0.7387 | 0.8824 | 3.2 | 0.025 | laptop |
| Gaussian NB | Probabilistic | 0.7555 ± 0.0198 | 0.7392 | 0.7008 | 0.8261 | 0.7351 | 0.8624 | 0.7623 | 0.7191 | 0.8275 | 0.7296 | 0.7561 | 1.9 | 0.017 | laptop |

Fit and inference times were measured on two different machines (laptop and Mac mini, last column), so they are only comparable within one machine.

### Training against test (overfitting check)

Each final model was refit on all 80 training trips and scored on those same trips. Accuracy and macro F1 are candidate-level; Top-1 is per GPS point.

| Model | Train accuracy | Train macro F1 | Train Top-1 | Test accuracy | Test macro F1 | Test Top-1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9347 | 0.8824 | 0.7411 | 0.9412 | 0.8939 | 0.8453 |
| Gaussian NB | 0.8202 | 0.7561 | 0.7303 | 0.8261 | 0.7623 | 0.8275 |
| KNN | 0.9509 | 0.9109 | 0.7782 | 0.9471 | 0.9037 | 0.8480 |
| Random Forest | 0.9752 | 0.9551 | 0.8322 | 0.9540 | 0.9162 | 0.8634 |
| LightGBM | 0.9592 | 0.9265 | 0.7864 | 0.9542 | 0.9172 | 0.8651 |
| MLP | 0.9510 | 0.9116 | 0.7654 | 0.9520 | 0.9133 | 0.8620 |

Random Forest has the largest gap (accuracy 0.9752 on training against 0.9540 on test, macro F1 0.9551 against 0.9162): mild overfitting. LightGBM, KNN and the MLP lose 0.4 accuracy points or less, so they show no real overfitting. Logistic Regression and Gaussian Naive Bayes score slightly higher on test than on training, which points to underfitting: their limit is the model's form, not the data.

Training Top-1 is lower than test Top-1 for every model. This is not underfitting: 11.9% of training points (mostly trip `00000054`) have no correct arc among their candidates and always count as wrong, against 0.4% on the student-test trips.

## 6. What we pick and why

**We take LightGBM forward** to Bayesian optimisation and the HMM decoder.

- **CV performance is a tie at the top.** Random Forest (0.8984), LightGBM (0.8969) and MLP (0.8930) are
  within 0.006 of each other on CV macro F1, while the fold-to-fold spread is about 0.027. CV alone cannot
  separate them, so we look at the other criteria.
- **Best on the student-test trips.** LightGBM has the highest student-test macro F1 (0.9172) and Top-1
  accuracy (0.8651) of the six models.
- **Best probabilities.** It has the lowest log loss on CV (0.1355, level with Random Forest's 0.1356) and
  on student test (0.1129 against 0.1153). The HMM uses these probabilities as emission scores, so
  calibrated probabilities matter more than a 0.5-threshold F1.
- **Less overfitting than Random Forest.** Train-to-test macro F1 gap: LightGBM 0.9265 to 0.9172, Random
  Forest 0.9551 to 0.9162. Random Forest fits the training trips much more closely without doing better on
  new trips.
- **Good target for tuning.** LightGBM has a small set of influential, continuous hyperparameters (learning
  rate, leaves, minimum child samples, regularisation, subsampling), which suits Bayesian search.

**What the others tell us.** Logistic Regression (0.8799 CV F1) is close behind, so most of the signal is in
the hand-built features rather than in non-linear interactions. KNN is competitive but the slowest to
predict and was fit on a 150k-row sample. Gaussian Naive Bayes is clearly worst (0.7555 CV F1, log loss
about 0.70): its independence assumption does not hold for strongly correlated features such as `dist`,
`rank` and `dist_minus_min`.

**Limits of this comparison.** All models are untuned. The margins between the top three are small, so the
choice rests on probability quality and stability as much as on accuracy. Classifier-only route continuity
is 0.74 to 0.78 for every model: picking each point's arc on its own often jumps between unconnected arcs,
which is what the HMM and Viterbi stage is for.
