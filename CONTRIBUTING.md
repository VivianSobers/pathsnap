# Contributing

Four people work on this repository: Vivian, Chirag, Harsimran and Chiranthan. This file holds the plan,
the task split and the working rules. ✅ means done and merged; ❌ means still to do.

## The problem in one paragraph

Each trip has a noisy 1 Hz GPS track (`.track`), its own road network (`.nodes`, `.arcs`) and a reference
route (`.route`, the arc ids the vehicle drove, in order). For every GPS observation we predict the road arc
it lies on. We generate candidate arcs near each observation, score every (observation, candidate) pair with
a classifier, then decode the whole trip with an HMM and Viterbi so consecutive arcs connect on the road
network. A FastAPI server on laptop 1 runs the pipeline and scores it against reference routes it keeps to
itself; a UI on laptop 2 lets the evaluator pick one of the ten reserved trips and shows the result.

## Progress at a glance

| Phase | Name | Owner(s) | Done | Left | Status |
|---|---|---|---|---|---|
| 0 | Repository and environment | Vivian | 1 | 2 | ❌ |
| 1 | Data loading and EDA | Harsimran, Vivian | 0 | 6 | ❌ |
| 2 | Labels and candidate generation | Vivian | 0 | 4 | ❌ |
| 3 | Feature engineering | Vivian, Chirag | 0 | 3 | ❌ |
| 4 | Model experiments (five or more models) | Chirag, Vivian | 0 | 7 | ❌ |
| 5 | Bayesian hyperparameter optimisation | Vivian | 0 | 3 | ❌ |
| 6 | HMM and Viterbi decoding | Vivian | 0 | 5 | ❌ |
| 7 | Folium maps and plots | Harsimran | 0 | 4 | ❌ |
| 8 | API | Chirag, Vivian | 0 | 4 | ❌ |
| 9 | UI | Chiranthan | 0 | 3 | ❌ |
| 10 | Integration, README and report | Everyone | 0 | 6 | ❌ |
| | **Total** | | **1** | **47** | |

## Share of work

| Person | Load | Main areas |
|---|---|---|
| Vivian | Highest | Labels, candidates, core features, final model, Bayesian optimisation, HMM/Viterbi, server-side evaluation, integration |
| Chirag | Medium to high | Metrics, cross-validation harness, three of the five models, comparison plots, FastAPI backend |
| Harsimran | Low to medium | EDA script, data-quality record, Folium maps, report figures |
| Chiranthan | Lowest | Evaluator UI, run instructions, UI and API screenshots |

Everyone must be able to explain their own phases in the viva, and should read the HMM section of the
report before evaluation day.

## Time plan

The first-phase evaluation is before the lunch break. Phases 0 to 4 (with at least the baseline models)
come first; the rest follows in order.

| Block | Target |
|---|---|
| Before lunch | Phases 0 to 3 done; Phase 4 with five models trained and compared on CV and student test |
| After lunch | Phases 5 and 6, then 7 to 9 in parallel |
| Last block | Phase 10: two-laptop rehearsal on all ten trips, README, report PDF |

## Phase 0: Repository and environment

| Status | Task | Owner | Where |
|---|---|---|---|
| ✅ | Repository, `.gitignore`, README stub | Vivian | `/` |
| ❌ | `requirements.txt` with pinned versions | Vivian | `requirements.txt` |
| ❌ | Package skeleton and config loader (paths, seeds, radii) | Vivian | `pathsnap/`, `configs/` |

## Phase 1: Data loading and EDA

Use the 80 training trips and the 10 student-testing trips only. State the partition on every table and plot.

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Trip loader for `.track`, `.nodes`, `.arcs`, `.route` | Vivian | `pathsnap/io.py` |
| ❌ | Projection to UTM zone 32N (EPSG:32632) for distances in metres | Vivian | `pathsnap/geo.py` |
| ❌ | Inventory table: trips, points, nodes, arcs, route length per partition | Harsimran | `scripts/eda.py`, `results/eda/` |
| ❌ | Sampling intervals, gaps over 5 s, very large gaps, missing or invalid values | Harsimran | `scripts/eda.py`, `results/eda/` |
| ❌ | Network connectivity: components, isolated nodes, route breaks | Harsimran | `scripts/eda.py`, `results/eda/` |
| ❌ | Data-quality record: issues, assumptions, decisions | Harsimran | `docs/data_quality.md` |

## Phase 2: Labels and candidate generation

The `.route` file has no timestamps, so each GPS observation needs a reference arc before we can train.

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Point-to-segment distance and projection on the arc | Vivian | `pathsnap/geo.py` |
| ❌ | Monotone alignment of observations to route arcs (dynamic programming over the route order) | Vivian | `pathsnap/labels.py` |
| ❌ | Candidate arcs within a radius using a spatial index, keep the k nearest | Vivian | `pathsnap/candidates.py` |
| ❌ | Candidate miss rate against radius and k, used to choose both | Vivian | `scripts/candidate_sweep.py`, `results/candidates/` |

## Phase 3: Feature engineering

One row per (observation, candidate) pair, with the trip id kept for grouped cross-validation.

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Geometric features: distance, distance rank, heading difference, arc length, position along arc | Vivian | `pathsnap/features.py` |
| ❌ | Motion and context features: speed, turn rate, gap since last point, distance to neighbours' candidates | Chirag | `pathsnap/features.py` |
| ❌ | Feature table build for train and student test, saved as Parquet | Vivian | `scripts/build_features.py` |

## Phase 4: Model experiments

All models use the same features, the same `GroupKFold` by trip on the 80 training trips, and the same seed.
Student-test trips are used only for comparison, never for fitting.

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Metrics: accuracy, precision, recall, Macro F1, log loss, Top-1, candidate miss rate, route continuity | Chirag | `pathsnap/metrics.py`, `tests/test_metrics.py` |
| ❌ | Cross-validation and experiment logging harness (config, seed, times, scores to JSON) | Chirag | `pathsnap/experiment.py` |
| ❌ | Logistic Regression and Gaussian Naive Bayes | Chirag | `scripts/train_models.py` |
| ❌ | KNN (distance-weighted) | Chirag | `scripts/train_models.py` |
| ❌ | Random Forest and Extra Trees | Vivian | `scripts/train_models.py` |
| ❌ | LightGBM and MLP | Vivian | `scripts/train_models.py` |
| ❌ | Comparison table and plots: CV Macro F1, student-test Top-1 and Macro F1, log loss, inference time | Chirag | `scripts/compare_models.py`, `results/models/` |

## Phase 5: Bayesian hyperparameter optimisation

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Optuna TPE search on the chosen model, grouped CV objective, search space in config | Vivian | `scripts/tune.py`, `configs/search_space.yaml` |
| ❌ | Trial record saved (every trial's parameters and score) | Vivian | `results/hpo/` |
| ❌ | Tuned against untuned on the same folds and student test | Vivian | `results/hpo/` |

## Phase 6: HMM and Viterbi decoding

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Hidden states are candidate arcs per observation; emissions from classifier probabilities | Vivian | `pathsnap/hmm.py` |
| ❌ | Transitions from network shortest-path distance against straight-line distance | Vivian | `pathsnap/hmm.py` |
| ❌ | Viterbi in log space; split decoding at large gaps; handle empty candidate sets and broken chains | Vivian | `pathsnap/hmm.py`, `tests/test_hmm.py` |
| ❌ | Decoder parameter tuning on training and student-test trips only | Vivian | `scripts/tune_decoder.py` |
| ❌ | Classifier-only against classifier plus HMM: Top-1, route continuity, sequence-level scores | Vivian | `results/hmm/` |

## Phase 7: Folium maps and plots

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Map of a trip: GPS points, nearby road network, reference route | Harsimran | `pathsnap/maps.py` |
| ❌ | Predicted route layer, legend, layer control, fitted zoom | Harsimran | `pathsnap/maps.py` |
| ❌ | Maps for selected train and student-test trips | Harsimran | `results/maps/` |
| ❌ | EDA plots for the report (coordinate spread, interval histogram, network size) | Harsimran | `scripts/eda.py`, `results/eda/` |

## Phase 8: API

Reference routes for the evaluation trips stay on the server and never go into a response or into the
prediction pipeline.

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Model and artifact export (classifier, preprocessing, HMM config) | Vivian | `scripts/export.py`, `artifacts/` |
| ❌ | FastAPI app: `GET /trips`, `POST /predict/{trip_id}` returning route and scores | Chirag | `api/` |
| ❌ | Server-side scoring against reference routes, kept apart from prediction code | Vivian | `api/evaluate.py` |
| ❌ | Request logging visible in the terminal; API tests | Chirag | `api/`, `tests/test_api.py` |

## Phase 9: UI

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | Page listing the ten evaluation trips, API address configurable | Chiranthan | `ui/` |
| ❌ | Show the returned route on a map and the scores | Chiranthan | `ui/` |
| ❌ | Clear error message when the API is unreachable | Chiranthan | `ui/` |

## Phase 10: Integration, README and report

| Status | Task | Owner | Where |
|---|---|---|---|
| ❌ | End-to-end run on all ten trips from a clean checkout | Vivian | `scripts/` |
| ❌ | Two-laptop rehearsal: API on laptop 1, UI on laptop 2, logs shown | Chiranthan, Chirag | |
| ❌ | README: install, run, deploy on two laptops | Chiranthan | `README.md` |
| ❌ | Screenshots of UI and API responses | Chiranthan | `report/figures/` |
| ❌ | Report sections: EDA (Harsimran), models (Chirag), HPO and HMM (Vivian) | Everyone | `report/` |
| ❌ | Final check that every number in the report matches a file in `results/` | Vivian | `report/` |

## Data and leakage rules

- Never copy the raw data into the repository; it lives in `data/raw/`, which git ignores.
- Split only by trip. No observations from one trip in two folds or two partitions.
- Fit scalers, encoders and any other learned preprocessing on training folds only.
- The ten evaluator trips are never used for training, feature selection, model choice, hyperparameters or
  decoder settings.
- No hardcoded predictions or trip-specific rules.

## Commits

- One logical change per commit. Tests pass at every commit.
- Subject line only, in the form `<type>: <description>`, with a 4 to 6 word description and no trailing period.
- Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`, `exp` (experiment results).
- No trailers.
- Commit as each piece lands; do not batch a day's work into one commit.

Good: `feat: add monotone route alignment`, `exp: add five model comparison results`
Bad: `update code`, `fix stuff`, `wip`

## Results

Never type a score into the report or README by hand. Every number comes from a file in `results/` written
by a script, with the seed, the config and the commit it ran on.
