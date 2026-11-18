"""Score candidate arcs with a fitted model, then decode the trip."""

import numpy as np

from pathsnap.features import FEATURES, build_trip_features, trip_geometry
from pathsnap.hmm import decode_route


def predict_trip(trip, artifact):
    if artifact["features"] != FEATURES:
        raise ValueError("inference artifact uses a different feature schema")
    if trip.route is not None:
        raise ValueError("prediction requires a trip without reference labels")
    if trip.track.empty:
        return np.empty(0, dtype=int)
    geometry = trip_geometry(trip)
    candidates, _ = build_trip_features(
        trip, **artifact["candidates"], gap_limit_s=artifact["split_s"], geometry=geometry,
    )
    if candidates.empty:
        return np.full(len(trip.track), -1, dtype=int)
    model = artifact["model"]
    inputs = np.nan_to_num(candidates[FEATURES].to_numpy(np.float64), nan=0, posinf=0, neginf=0)
    probabilities = model.predict_proba(inputs)[:, list(model.classes_).index(1)]
    states = candidates.rename(columns={
        "obs": "observation_id", "arc": "arc_id", "t_along": "position_along_arc",
    })
    return decode_route(
        trip, geometry, states, probabilities, split_s=artifact["split_s"], beta_m=artifact["beta_m"],
    )
