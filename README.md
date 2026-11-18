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
python scripts/build_features.py                  # writes data/interim/ and results/candidates/
python scripts/train_models.py --model lightgbm   # one run per model, writes results/models/<model>.json
python scripts/compare_models.py                  # writes results/models/comparison.{csv,md}
```

Model names: `logreg`, `gaussian_nb`, `knn`, `random_forest`, `lightgbm`, `mlp`.

## Run on two laptops

Laptop 1 runs the API, which holds the reference routes. Laptop 2 runs the evaluator UI. Both laptops must
be on the same network.

### Laptop 1: API

1. Find its IP address: `ipconfig` on Windows (the IPv4 address of the active adapter).
2. Start the server so that other machines can reach it:

   ```
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   > The API is not implemented yet (`api/main.py` is a stub); this command is the expected one and must be
   > confirmed once it lands.
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
3. If the UI shows "Cannot reach the API" but the step above works, the API is missing CORS headers.

## Repository layout

| Path | Purpose |
|---|---|
| `pathsnap/` | Library: loading, geometry, labels, candidates, features, metrics, experiments |
| `scripts/` | Command-line steps of the pipeline |
| `api/` | FastAPI server and server-side scoring |
| `ui/` | Evaluator page (`ui/mock_api.py` is a stand-in for testing the UI alone) |
| `results/` | Every number quoted in the report is written here by a script |
| `CONTRIBUTING.md` | Plan, task split and working rules |
