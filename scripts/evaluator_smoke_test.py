"""Check that the API returns a valid route for every evaluator trip, without scoring anything.

The scorer is replaced by a stub, so reference routes are never read and no accuracy is computed.
Writes results/e2e/evaluator_smoke_test.json (status, point counts, route segments, runtime).

    python -m scripts.evaluator_smoke_test artifacts/<model>.joblib data/raw/evaluator/evaluator_test
"""

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from joblib import load

from api.main import create_app
from pathsnap.predict import predict_trip


def main():
    artifact_path, trips_dir = sys.argv[1], Path(sys.argv[2])
    art = load(artifact_path)
    client = TestClient(create_app(trips_dir, predict=lambda t: predict_trip(t, art), score=lambda d, a: {}))
    rows = []
    for t in client.get("/trips").json()["trips"]:
        r = client.post(f"/predict/{t['trip_id']}")
        j = r.json() if r.status_code == 200 else {}
        rows.append({"trip": t["trip_id"], "status": r.status_code, "n_points": t["n_points"],
                     "predicted_segments": len(j.get("route", [])), "runtime_s": j.get("runtime_s"),
                     "scores_returned": j.get("scores")})
    out = {"artifact": artifact_path, "scoring": "disabled (stub)", "trips": rows,
           "valid": sum(r["status"] == 200 and r["predicted_segments"] > 0 for r in rows)}
    Path("results/e2e/evaluator_smoke_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
