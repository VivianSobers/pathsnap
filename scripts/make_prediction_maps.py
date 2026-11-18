"""Folium maps with GPS points, road network, reference route and the predicted route.

    python scripts/make_prediction_maps.py artifacts/<model>.joblib student_test 00000007 00000077

The prediction is made from the trip without its route file (as in the API); the reference route is
loaded separately, only for drawing. Writes results/maps/<partition>_<trip>_predicted.html.
"""

import sys
from pathlib import Path

import numpy as np
from joblib import load

from pathsnap.config import load_config
from pathsnap.eda_io import load_trip as load_map_trip
from pathsnap.io import load_trip
from pathsnap.maps import build_trip_map
from pathsnap.predict import predict_trip


def main():
    artifact_path, part, *trip_ids = sys.argv[1:]
    artifact = load(artifact_path)
    cfg = load_config()
    out = cfg["paths"]["results"] / "maps"; out.mkdir(parents=True, exist_ok=True)
    for tid in trip_ids:
        d = cfg["paths"][part] / tid
        arcs = predict_trip(load_trip(d, include_route=False), artifact)
        arcs = arcs[arcs >= 0]
        ordered = arcs[np.r_[True, arcs[1:] != arcs[:-1]]].tolist()  # drop consecutive repeats
        m = build_trip_map(load_map_trip(d, part), pred_arcs=ordered,
                           title=f"{part} / trip {tid} / {Path(artifact_path).stem}")
        path = out / f"{part}_{tid}_predicted.html"
        m.save(str(path)); print("wrote", path)


if __name__ == "__main__":
    main()
