# PathSnap: phase 2 report

Candidate tuning, Bayesian hyperparameter optimisation, HMM/Viterbi route reconstruction, a feature
ablation and the first end-to-end API run. Every number is copied from the file named with its table.
Partitions: CV = 5-fold GroupKFold by trip on the 80 training trips; test = the 10 student-test trips.
Evaluator trips were not used for anything.

## 1. Candidate radius and k

From `results/candidates/sweep.csv` (`scripts/candidate_sweep.py`, no model involved).

| Radius (m) | k | Train miss rate | Train miss rate without trip 00000054 | Candidates per observation (train) | Student-test miss rate |
|---|---|---|---|---|---|
| 30 | 8 | 12.88% | 4.92% | 4.19 | 1.00% |
| **50** | **8** | **11.92%** | **3.87%** | **5.21** | **0.42%** |
| 50 | 12 | 11.66% | 3.60% | 6.44 | 0.21% |
| 75 | 12 | 11.18% | 3.12% | 7.84 | 0.18% |
| 150 | 16 | 6.93% | 2.74% | 12.96 | 0.08% |

Most of the training miss rate comes from one trip, `00000054`, whose GPS points mostly lie more than
50 m from its road network. Without it, the miss rate falls to 3.87% at 50 m / k=8. Larger settings
lower it only slowly while growing the candidate set (and every model's training data) quickly. We kept
50 m / k=8, chosen on the training trips.

## 2. Bayesian hyperparameter optimisation (LightGBM)

- **Method.** Optuna TPE sampler, seed 42 (worker *i* uses seed 42 + *i*), 8 random start-up trials, three
  workers sharing one study through a journal file (`scripts/tune.py`).
- **Objective.** Mean macro F1 over the same 5 GroupKFold-by-trip folds used in the model comparison,
  training trips only.
- **Search space** (`configs/search_space.yaml`): n_estimators 100 to 800; learning_rate 0.01 to 0.2 (log);
  num_leaves 15 to 255 (log); min_child_samples 5 to 200 (log); subsample 0.5 to 1.0; colsample_bytree 0.5
  to 1.0; reg_lambda and reg_alpha 0.001 to 10 (log).
- **Trials.** 32 completed (`results/hpo/trials.csv`). Three trials failed at the first launch because
  LightGBM could not load its OpenMP library on the Mac mini; one of them was the queued untuned
  configuration. The untuned baseline therefore comes from the separate run in `results/models/lightgbm.json`
  on the same folds.
- **Best trial** (`results/hpo/best_params.json`): trial 26, n_estimators 277, learning_rate 0.0369,
  num_leaves 118, min_child_samples 44, subsample 0.807, colsample_bytree 0.529, reg_lambda 0.0061,
  reg_alpha 0.0275. The best trial took 29.7 s on the Mac mini (5 folds).

Tuned against untuned, 12 features, from `results/models/lightgbm.json` and `lightgbm_tuned.json`:

| | CV macro F1 | CV log loss | Test accuracy | Test macro F1 | Test Top-1 | Test log loss |
|---|---|---|---|---|---|---|
| Untuned | 0.8969 ± 0.0276 | 0.1355 | 0.9542 | **0.9172** | **0.8651** | **0.1129** |
| Tuned | **0.9000** ± 0.0251 | **0.1332** | 0.9534 | 0.9157 | 0.8617 | 0.1155 |

**Did tuning help? No, not measurably.** The tuned model is 0.003 better on CV macro F1 and slightly worse
on every student-test metric. Both gaps are far smaller than the fold-to-fold standard deviation (about
0.025), so we read this as no real difference: the untuned defaults were already close to the best region
of the search space. Inference times are not compared here because the two runs were on different
machines (laptop and Mac mini).

## 3. Feature ablation: 12 against 15 features

Three motion and context features were added after phase 1: turn rate, and the distance from the
candidate's projected point to the nearest candidate of the previous and of the next observation. Untuned
LightGBM, same folds, from `results/models/lightgbm.json` and `lightgbm_15feat.json`:

| Features | CV macro F1 | CV log loss | Test macro F1 | Test Top-1 | Test log loss |
|---|---|---|---|---|---|
| 12 | 0.8969 | 0.1355 | 0.9172 | 0.8651 | 0.1129 |
| 15 | **0.8980** | **0.1315** | **0.9178** | **0.8667** | **0.1098** |

Every metric improves a little and none by more than the fold-to-fold spread. The largest relative gain is
in log loss, which is the quantity the HMM consumes. We keep the 15 features.

## 4. HMM and Viterbi route reconstruction

**Hidden states.** At each GPS observation, its candidate arcs (up to 8 within 50 m). Each state keeps its
arc id and its projected position on the arc, so the decoded state maps straight back to a road segment.

**Emissions.** w × log P(candidate is the reference arc), from LightGBM's probability, clipped at 1e-6.

**Transitions.** From candidate *a* at observation *i* to candidate *b* at observation *i+1*:
−|network distance − straight-line distance| / β. The network distance runs along the directed road graph
from the projected point on *a* to the projected point on *b* (Dijkstra, capped at 3 km); staying on the
same arc uses the distance along it. Pairs that cannot be reached get −∞.

**Gaps and disconnections.** Decoding restarts at time gaps over 60 s and wherever every transition between
two consecutive observations is −∞ (the candidate roads do not connect). Observations without candidates
get no prediction and are bridged.

**Decoder parameters.** Only the product β × w matters: scaling the whole Viterbi score by w leaves the best
path unchanged, so (β, w) = (20, 5), (50, 2) and (100, 1) give identical results. We searched β over
2 to 1000 and w over 0.5 to 8 using out-of-fold probabilities on the training trips only
(`results/hmm/hmm_vs_classifier_wide.json`); training Top-1 peaks at β × w = 100.

Classifier-only against classifier + HMM, untuned LightGBM, 12 features, from
`results/hmm/hmm_vs_classifier.json`:

| | Train (out-of-fold) classifier | Train + HMM | Test classifier | Test + HMM |
|---|---|---|---|---|
| Top-1 accuracy | 0.7514 | 0.7666 | 0.8651 | **0.8877** |
| Route continuity | 0.7573 | 0.8540 | 0.7760 | **0.8691** |
| Route arc precision | 0.8094 | 0.8633 | 0.8328 | **0.8901** |
| Route arc recall | 0.8837 | 0.8945 | 0.8850 | **0.8958** |
| Route arc F1 | 0.8367 | 0.8717 | 0.8548 | **0.8902** |

The HMM improves every metric on both partitions. The largest gain is route continuity (+9.3 points on
test): the classifier alone often jumps between arcs that do not connect, and the transition model rules
most of those jumps out. Route arc precision rises because spurious one-point detours onto side streets
disappear.

**Where the HMM did not help.** Continuity and Top-1 pull against each other: a small β × w forces smoother
routes (training continuity 0.887 at β × w = 1) but lowers Top-1 (0.713), because the decoder then
overrides correct classifier choices to avoid short legitimate detours. We select on Top-1. Continuity
cannot reach 100% in any case, because some reference routes contain breaks themselves.

## 5. End-to-end API run

The exported model (untuned LightGBM, 15 features, β = 100, `scripts/export_model.py`) served by the
FastAPI app with the 10 student-test trips as the trip directory, every trip requested through
`POST /predict/{trip_id}` (`results/e2e/api_student_test_lightgbm_f15_untuned.json`):

| Trip | Points | Top-1 | Route continuity | Runtime (s) |
|---|---|---|---|---|
| 00000000 | 2503 | 0.9565 | 0.8987 | 1.88 |
| 00000007 | 1354 | 0.9830 | 0.9851 | 0.95 |
| 00000009 | 1049 | 0.8818 | 0.8232 | 1.08 |
| 00000040 | 2995 | 0.9329 | 0.9107 | 2.46 |
| 00000052 | 262 | 0.9924 | 0.9036 | 0.24 |
| 00000056 | 3958 | 0.9224 | 0.8649 | 2.15 |
| 00000065 | 768 | 0.8477 | 0.8486 | 0.45 |
| 00000076 | 1486 | 0.8970 | 0.7931 | 1.00 |
| 00000077 | 3911 | 0.7387 | 0.9125 | 1.08 |
| 00000091 | 874 | 0.8627 | 0.7811 | 0.31 |

All 10 requests returned HTTP 200. Pooled Top-1 over all 19,160 points is 0.8863 and mean route continuity
is 0.8722, in line with the offline evaluation. The weakest trip is `00000077` (Top-1 0.7387) despite high
continuity, which suggests the decoder follows a consistent but wrong parallel road for part of that trip;
it is the first candidate for a map-based error analysis.
