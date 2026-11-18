"""Minimal raw-data reader used by the EDA and map code (Harsimran).

Vivian owns the real loader in pathsnap/io.py. Once that is merged, replace the
body of load_trip() with a call to it. The file format is from the Dataset
Exploration PDF:
  .track  lon lat time_s   (whitespace separated, no header)
  .nodes  lon lat          (row index = node_id)
  .arcs   from_node to_node (row index = arc_id)
  .route  arc_id           (reference route, in order)
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

RAW_ROOT = Path("data/raw")
PARTITIONS = ("train", "student_test")   # evaluator trips are never touched


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray   # (n, 3) lon, lat, t
    nodes: np.ndarray   # (m, 2) lon, lat
    arcs: np.ndarray    # (a, 2) int from_node, to_node
    route: np.ndarray   # (r,) int arc ids


def trip_dirs(partition, root=RAW_ROOT):
    return sorted(p for p in (Path(root) / partition).iterdir() if p.is_dir())


def load_trip(trip_dir, partition):
    d = Path(trip_dir)
    t = d.name
    rd = lambda ext, **kw: np.loadtxt(d / f"{t}.{ext}", ndmin=2, **kw)
    return Trip(
        trip_id=t,
        partition=partition,
        track=rd("track"),
        nodes=rd("nodes"),
        arcs=rd("arcs", dtype=int),
        route=np.loadtxt(d / f"{t}.route", dtype=int, ndmin=1),
    )


def iter_trips(partitions=PARTITIONS, root=RAW_ROOT):
    for part in partitions:
        for d in trip_dirs(part, root):
            yield load_trip(d, part)


def to_local_metres(lon, lat, lon0, lat0):
    """Local equirectangular projection (metres) around (lon0, lat0).

    Trips are in different cities all over the world, so ONE fixed UTM zone
    cannot be used. Per trip, this is accurate to well under 1% at city scale.
    """
    R = 6371008.8
    x = np.radians(np.asarray(lon) - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R
    return x, y
