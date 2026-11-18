# PathSnap evaluator UI

A static page (`index.html`, Leaflet from a CDN, no build step). It runs on laptop 2 and talks to the API on
laptop 1. It needs internet access for the Leaflet files and the OpenStreetMap tiles.

## Run

```
python -m http.server 8080 --directory ui
```

Open `http://localhost:8080`, type the API address (for example `http://192.168.1.10:8000`) and press
Connect. The address is remembered, and can also be given as `?api=http://host:port`.

## API contract the UI expects

`GET /trips`

```json
{"trips": [{"trip_id": "abc", "n_points": 412}]}
```

`POST /predict/{trip_id}` (the response never contains the reference route)

```json
{"trip_id": "abc",
 "gps": [[lat, lon], ...],
 "route": [[[lat, lon], [lat, lon]], ...],
 "scores": {"top1_accuracy": 0.0, "route_continuity": 0.0},
 "runtime_s": 0.0}
```

- `route` is a list of polylines, one per predicted arc (or per run of arcs), in `[lat, lon]` order.
- `scores` is shown as a name/value table, so any metric names work.
- The API must send CORS headers (`CORSMiddleware`, allow all origins) because the page is served from a
  different origin.

## Test without the real API

`python ui/mock_api.py` serves synthetic geometry and placeholder zeros. It is for checking the UI only.
