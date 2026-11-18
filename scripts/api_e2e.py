"""Run every trip in a folder through the API with server-side scoring and save the scores.

Uses the same app as `uvicorn api.main:app` (artifact, decoder and scorer from the environment setup),
called in-process. Writes results/e2e/api_<partition>_<artifact>.json with per-trip Top-1, route
continuity and runtime, the pooled Top-1 (weighted by points), the mean continuity and the commit.

    python -m scripts.api_e2e artifacts/<model>.joblib data/raw/<...>/<partition_dir>
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient


def main():
    artifact_path, trips_dir = sys.argv[1], Path(sys.argv[2])
    os.environ["PATHSNAP_MODEL_ARTIFACT"] = artifact_path
    os.environ["PATHSNAP_EVALUATION_DIR"] = str(trips_dir)
    from api.main import create_configured_app

    client = TestClient(create_configured_app())
    rows = []
    for t in client.get("/trips").json()["trips"]:
        r = client.post(f"/predict/{t['trip_id']}")
        r.raise_for_status()
        j = r.json()
        rows.append({"trip": t["trip_id"], "n_points": t["n_points"], **j["scores"], "runtime_s": j["runtime_s"]})
    n = sum(r["n_points"] for r in rows)
    out = {
        "artifact": artifact_path,
        "partition": trips_dir.name,
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trips": rows,
        "pooled_top1_accuracy": sum(r["top1_accuracy"] * r["n_points"] for r in rows) / n,
        "mean_route_continuity": sum(r["route_continuity"] for r in rows) / len(rows),
    }
    path = Path("results/e2e") / f"api_{trips_dir.name}_{Path(artifact_path).stem}.json"
    path.write_text(json.dumps(out, indent=2))
    print(path)
    for r in rows:
        print(f"{r['trip']}  n={r['n_points']:5d}  top1={r['top1_accuracy']:.4f}  continuity={r['route_continuity']:.4f}  {r['runtime_s']:.2f}s")
    print(f"pooled top1 {out['pooled_top1_accuracy']:.4f}   mean continuity {out['mean_route_continuity']:.4f}")


if __name__ == "__main__":
    main()
