"""Score candidate arcs with a fitted model, then decode the trip."""

import numpy as np

from pathsnap.features import FEATURES, build_trip_features, trip_geometry
from pathsnap.geo import local_origin, to_metric
from pathsnap.hmm import decode_trip


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
    origin = local_origin(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    nx, ny = to_metric(trip.nodes["lon"], trip.nodes["lat"], origin)
    px, py = geometry[0].T
    chosen = decode_trip(
        candidates["obs"].to_numpy(), candidates["arc"].to_numpy(),
        candidates["t_along"].to_numpy(), probabilities, px, py, trip.track["t"].to_numpy(),
        nx, ny, trip.arcs[["from_node", "to_node"]].to_numpy(),
        split_s=artifact["split_s"], beta=artifact["beta_m"],
        uturn_penalty=artifact.get("uturn_penalty", 0.0),
    )
    output = np.full(len(trip.track), -1, dtype=int)
    for observation, arc in chosen.items():
        output[observation] = arc
    return output
