"""Server-side scoring of predicted routes against reference routes."""

import numpy as np

from pathsnap.features import trip_geometry
from pathsnap.io import load_trip
from pathsnap.labels import align_to_route

def score_trip(directory, arc_ids):
    """Align and score private reference labels after inference."""
    trip = load_trip(directory)
    predicted = np.asarray(arc_ids, dtype=int)
    accuracy = None
    if trip.route is not None and len(trip.route) and len(predicted):
        points, starts, ends = trip_geometry(trip)
        px, py = points.T
        ax, ay = starts[trip.route].T
        bx, by = ends[trip.route].T
        reference = trip.route[align_to_route(px, py, ax, ay, bx, by)]
        accuracy = float(np.mean(predicted == reference))
    adjacent = (predicted[:-1] >= 0) & (predicted[1:] >= 0)
    changes = adjacent & (predicted[:-1] != predicted[1:])
    continuity = None
    if changes.any():
        arcs = trip.arcs.to_numpy()
        continuity = float(np.mean(arcs[predicted[:-1][changes], 1] == arcs[predicted[1:][changes], 0]))
    return {"top1_accuracy": accuracy, "route_continuity": continuity}
