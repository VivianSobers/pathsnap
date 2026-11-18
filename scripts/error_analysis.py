"""Break down student-test errors of a decoded run by type, and check reference-arc direction.

    python scripts/error_analysis.py results/hmm/student_test_predictions_f15.json

Writes results/hmm/error_analysis_<name>.json. Error types, per GPS observation:
  reverse_direction  predicted arc is the reference arc's twin in the opposite direction
  other_route_arc    predicted arc is on the reference route, but at another position
  adjacent_off_route predicted arc is off the route and shares a node with the reference arc
  other_off_route    any other wrong arc;  no_prediction: observation had no candidates
Also reports how often the reference arc points against the vehicle's motion (speed > 2 m/s).
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.config import load_config
from pathsnap.io import load_trip


def main():
    pred_path = Path(sys.argv[1])
    cfg = load_config()
    pred = json.loads(pred_path.read_text())
    ref = json.loads((cfg["paths"]["interim"] / "obs_student_test.json").read_text())
    feats = pd.read_parquet(cfg["paths"]["interim"] / "features_student_test.parquet",
                            columns=["trip", "obs", "arc", "cos_heading", "speed", "label"])
    cos = feats.set_index(["trip", "obs", "arc"])["cos_heading"]
    counts, rev_against, per_trip = Counter(), [], {}
    for tid, r in ref.items():
        trip = load_trip(cfg["paths"]["student_test"] / tid)
        arcs, route = trip.arcs.to_numpy(), set(trip.route.tolist())
        c = Counter()
        for i, ra in enumerate(r):
            pa = pred[tid].get(str(i))
            if pa is None:
                k = "no_prediction"
            elif pa == ra:
                k = "correct"
            elif arcs[pa][0] == arcs[ra][1] and arcs[pa][1] == arcs[ra][0]:
                k = "reverse_direction"
                rev_against.append(cos.get((tid, i, ra), np.nan) < -0.5)
            elif pa in route:
                k = "other_route_arc"
            elif {*arcs[pa]} & {*arcs[ra]}:
                k = "adjacent_off_route"
            else:
                k = "other_off_route"
            c[k] += 1
        counts.update(c); per_trip[tid] = dict(c)
    n = sum(counts.values())
    moving = feats[(feats.label == 1) & (feats.speed > 2)]
    out = {
        "predictions": str(pred_path), "observations": n,
        "share": {k: v / n for k, v in counts.most_common()},
        "reverse_errors_with_reference_against_motion": float(np.mean(rev_against)) if rev_against else None,
        "reference_arc_against_motion_share_moving_points": float((moving.cos_heading < 0).mean()),
        "per_trip": per_trip,
    }
    dest = pred_path.with_name("error_analysis_" + pred_path.stem.replace("student_test_predictions", "").strip("_") + ".json")
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "per_trip"}, indent=2))


if __name__ == "__main__":
    main()
