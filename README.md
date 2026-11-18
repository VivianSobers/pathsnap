# PathSnap

Map-matching and route reconstruction for noisy GPS trajectories. Each GPS observation is matched to a
road segment using a candidate classifier followed by HMM/Viterbi decoding, served through a FastAPI
backend and an evaluator UI.

## Requirements

- Python 3.11 or newer (developed on 3.12)
- The dataset, placed in `data/raw/Student folder/` with `train/` and `student_test/` inside (git ignores
  `data/`, never commit it)

## Install

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux or macOS
pip install -r requirements.txt
```

Run the tests with `python -m pytest`.

## Build features and train models

All settings (seed, paths, candidate radius, folds) are in `configs/default.yaml`.

```
python -m scripts.build_features                  # writes data/interim/ and results/candidates/
python -m scripts.train_models --model lightgbm   # one run per model, writes results/models/<model>.json
python -m scripts.compare_models                  # writes results/models/comparison.{csv,md,png}
```

Model names: `logreg`, `gaussian_nb`, `knn`, `random_forest`, `lightgbm`, `mlp`.

## Run on two laptops

Laptop 1 runs the API, which holds the reference routes. Laptop 2 runs the evaluator UI. Both laptops must
be on the same network.

### Laptop 1: API

1. Find its IP address: `ipconfig` on Windows (the IPv4 address of the active adapter).
2. Point the API to the final model artifact (committed in `artifacts/`), the evaluator-supplied trip
   directory, and the actual origin of the UI open on laptop 2. To rebuild the artifact from the
   training data instead: `python -m scripts.export_model --jobs 8 --params-json
   results/hpo/best_params_lightgbm_f15.json --output artifacts/lightgbm_f15_tuned_uturn10.joblib`.

   ```powershell
   $env:PATHSNAP_MODEL_ARTIFACT = (Resolve-Path artifacts/lightgbm_f15_tuned_uturn10.joblib).Path
   $env:PATHSNAP_EVALUATION_DIR = 'C:\path\to\reserved-trips'
   $env:PATHSNAP_UI_ORIGIN = 'http://localhost:8080'
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   On Linux or macOS (find the IP with `ip -4 addr` or `ipconfig getifaddr en0`):

   ```bash
   export PATHSNAP_MODEL_ARTIFACT="$PWD/artifacts/lightgbm_f15_tuned_uturn10.joblib"
   export PATHSNAP_EVALUATION_DIR=/path/to/reserved-trips
   export PATHSNAP_UI_ORIGIN=http://<laptop-2-ip>:8080
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   For development only, use `data/raw/Student folder/student_test` as the trip directory.
3. Allow the port through the firewall (Windows, run once as administrator):

   ```
   netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
   ```

Requests are logged in this terminal; keep it visible during the evaluation.

### Laptop 2: UI

```
python -m http.server 8080 --directory ui
```

Open `http://localhost:8080`, type `http://<laptop-1-ip>:8000` in the API address box and press Connect.
Pick one of the ten trips to run the prediction. The page needs internet access for the map library and
tiles. The API contract the UI expects is in [ui/README.md](ui/README.md).

### Quick check before the evaluation

1. From laptop 2, open `http://<laptop-1-ip>:8000/trips` in a browser. A JSON list means the network path
   works.
2. If it times out, check the firewall rule and that both laptops are on the same network.
3. If the UI shows "Cannot reach the API" but the step above works, check that
   `PATHSNAP_UI_ORIGIN` matches the UI URL, including its scheme, hostname and port.

## Reproduce the report

Every number and figure in the technical report is read from `results/` by `report/build_report.py`. The
commands below rewrite those files. They use the 15-feature set now in `pathsnap/features.py`; the
12-feature model comparison (report section 3) was produced at commit `78f6c4e`.

| Report section | Command | Output |
|---|---|---|
| 2. EDA | `python scripts/eda.py` and `python -m scripts.road_network_plot` | `results/eda/` |
| 2. Candidates and features | `python -m scripts.build_features` and `python -m scripts.candidate_sweep` | `data/interim/`, `results/candidates/` |
| 3. Models | `python -m scripts.train_models --model <name>` for each model, then `python -m scripts.compare_models` | `results/models/` |
| 4.2 Untuned LightGBM, 15 features | `python -m scripts.train_models --model lightgbm --name lightgbm_15feat` | `results/models/lightgbm_15feat.json` |
| 3.3 Diagnostics | `python -m scripts.model_diagnostics` and `python -m scripts.time_inference` | `results/models/diagnostics/`, `results/models/inference_time.json` |
| 4. Bayesian optimisation | `python scripts/tune.py search --study lightgbm_f15 --worker <i>` (one or more workers), then `python scripts/tune.py finalize --study lightgbm_f15 --name lightgbm_f15_tuned` | `results/hpo/`, `results/models/lightgbm_f15_tuned.json` |
| 5. HMM | `python -m scripts.hmm_eval --betas 50 100 200 --ws 1 --uturns 0 2 5 10 --tag _f15` (add `--exclude-train-trips 00000054 --tag _f15_excl54` for the trip-54 check) | `results/hmm/` |
| 5. Error analysis | `python scripts/error_analysis.py results/hmm/student_test_predictions_f15.json` | `results/hmm/error_analysis_f15.json` |
| 6. Folium maps | `python scripts/make_prediction_maps.py artifacts/lightgbm_f15_tuned_uturn10.joblib student_test 00000007 00000009 00000077` | `results/maps/` |
| 7. End to end | `python -m scripts.api_e2e artifacts/lightgbm_f15_tuned_uturn10.joblib "data/raw/Student folder/student_test"` (or the evaluator trip directory) | `results/e2e/` |
| Report | `python report/build_report.py`, then `python report/build_pdf.py` (needs Chrome or Chromium) | `report/PathSnap_Report.md`, `report/PathSnap_Report.pdf` |

## Repository layout

| Path | Purpose |
|---|---|
| `pathsnap/` | Library: loading, geometry, labels, candidates, features, metrics, experiments |
| `scripts/` | Command-line steps of the pipeline |
| `api/` | FastAPI server and server-side scoring |
| `ui/` | Evaluator page (`ui/mock_api.py` is a stand-in for testing the UI alone) |
| `results/` | Every number quoted in the report is written here by a script |
| `CONTRIBUTING.md` | Plan, task split and working rules |
