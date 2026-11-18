# Report section draft: UI and deployment (Chiranthan)

Fill the bracketed items from real runs. Do not type in scores; quote files in `results/`.

## Evaluator UI

- Static page served from laptop 2; talks to the API on laptop 1 over HTTP (`GET /trips`, `POST /predict/{trip_id}`).
- What the evaluator sees: trip list, map with GPS points and predicted route, score table, error banner.
- Design choices to explain: [why a static page, why the API address is configurable, why no reference route is ever sent to the UI].
- Failure handling: unreachable server, HTTP error, non-JSON reply each give a plain message.

## Deployment on two laptops

- Topology: API on laptop 1 (Fedora) on port 8000, UI on laptop 2 (Windows) on port 8080.
- Network used in the rehearsal: campus Wi-Fi; laptop 1 at `10.30.202.250`, laptop 2 at `10.30.200.139`.
- Steps: see README, "Run on two laptops".

## Rehearsal log

| Date | Trips run | Problems found | Fix |
|---|---|---|---|
| 2026-11-18 | Trip list (10 evaluator trips), predict on 00000023 | UI said "Cannot reach the API" although requests reached laptop 1: the page was not opened from the origin the API allows (CORS) | Open the UI at laptop 2's network address, `http://10.30.200.139:8080`, matching `PATHSNAP_UI_ORIGIN` |

## Figures (in `report/figures/`)

- [x] `ui_connectivity.png` (address bar shows UI on laptop 2, API on laptop 1)
- [x] `ui_trip_list.png`
- [x] `ui_route_and_scores.png`
- [x] `api_terminal_trips.png`
- [x] `api_terminal_predict.png` (request log including `POST /predict/00000023` from laptop 2)
