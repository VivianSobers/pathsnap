from pathlib import Path

import numpy as np
import pytest

from pathsnap.io import list_trips, load_trip


def write_trip(root: Path, trip_id: str, route: bool = True) -> Path:
    d = root / trip_id
    d.mkdir(parents=True)
    (d / f"{trip_id}.track").write_text("8.60 49.80 1\n8.61 49.81 2\n8.62 49.82 4\n")
    (d / f"{trip_id}.nodes").write_text("8.60 49.80\n8.61 49.81\n8.62 49.82\n")
    (d / f"{trip_id}.arcs").write_text("0 1\n1 2\n")
    if route:
        (d / f"{trip_id}.route").write_text("0\n1\n")
    return d


def test_load_trip_reads_every_file(tmp_path):
    write_trip(tmp_path, "00000003")
    trip = load_trip(tmp_path / "00000003")
    assert trip.trip_id == "00000003"
    assert list(trip.track.columns) == ["lon", "lat", "t"]
    assert len(trip.track) == 3 and trip.track["t"].tolist() == [1, 2, 4]
    assert list(trip.nodes.columns) == ["lon", "lat"] and len(trip.nodes) == 3
    assert list(trip.arcs.columns) == ["from_node", "to_node"]
    assert trip.arcs["from_node"].dtype.kind == "i"
    np.testing.assert_array_equal(trip.route, [0, 1])


def test_missing_route_is_none(tmp_path):
    write_trip(tmp_path, "00000004", route=False)
    assert load_trip(tmp_path / "00000004").route is None


def test_arc_pointing_past_last_node_is_rejected(tmp_path):
    d = write_trip(tmp_path, "00000005")
    (d / "00000005.arcs").write_text("0 1\n1 9\n")
    with pytest.raises(ValueError, match="node"):
        load_trip(d)


def test_route_pointing_past_last_arc_is_rejected(tmp_path):
    d = write_trip(tmp_path, "00000006")
    (d / "00000006.route").write_text("0\n5\n")
    with pytest.raises(ValueError, match="arc"):
        load_trip(d)


def test_list_trips_is_sorted(tmp_path):
    for t in ["00000009", "00000001"]:
        write_trip(tmp_path, t)
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in list_trips(tmp_path)] == ["00000001", "00000009"]
