# Report section draft: UI and deployment (Chiranthan)

Fill the bracketed items from real runs. Do not type in scores; quote files in `results/`.

## Evaluator UI

- Static page served from laptop 2; talks to the API on laptop 1 over HTTP (`GET /trips`, `POST /predict/{trip_id}`).
- What the evaluator sees: trip list, map with GPS points and predicted route, score table, error banner.
- Design choices to explain: [why a static page, why the API address is configurable, why no reference route is ever sent to the UI].
- Failure handling: unreachable server, HTTP error, non-JSON reply each give a plain message.

## Deployment on two laptops

- Topology: [diagram or sentence], ports [8000 API, 8080 UI].
- Network setup used on the day: [network name, laptop 1 IP].
- Steps: see README, "Run on two laptops".

## Rehearsal log

| Date | Trips run | Problems found | Fix |
|---|---|---|---|
| [ ] | [ ] | [ ] | [ ] |

## Figures (in `report/figures/`)

- [ ] `ui_trip_list.png`
- [ ] `ui_route_and_scores.png`
- [ ] `ui_api_unreachable.png`
- [ ] `api_terminal_logs.png`
- [ ] `api_predict_response.png`
