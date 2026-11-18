"""Read a trip's .track, .nodes, .arcs and .route files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame  # lon, lat, t (seconds)
    nodes: pd.DataFrame  # lon, lat; row index is the node id
    arcs: pd.DataFrame  # from_node, to_node; row index is the arc id
    route: np.ndarray | None  # reference arc ids in driving order, if supplied


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(trip_dir: str | Path) -> Trip:
    """Load one trip folder and check that arc and route ids point at existing rows."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    base = trip_dir / tid

    track = _read(base.with_suffix(".track"), ["lon", "lat", "t"], float)
    nodes = _read(base.with_suffix(".nodes"), ["lon", "lat"], float)
    arcs = _read(base.with_suffix(".arcs"), ["from_node", "to_node"], np.int64)

    route_path = base.with_suffix(".route")
    route = None
    if route_path.exists():
        route = _read(route_path, ["arc"], np.int64)["arc"].to_numpy()

    n_nodes = len(nodes)
    if len(arcs) and (arcs.to_numpy().min() < 0 or arcs.to_numpy().max() >= n_nodes):
        raise ValueError(f"trip {tid}: an arc refers to a node id outside 0..{n_nodes - 1}")
    if route is not None and len(route) and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"trip {tid}: the route refers to an arc id outside 0..{len(arcs) - 1}")

    return Trip(tid, track, nodes, arcs, route)


def list_trips(partition_dir: str | Path) -> list[Path]:
    """Trip folders in a partition, sorted by trip id."""
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())
