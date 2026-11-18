import logging

from fastapi.testclient import TestClient
from joblib import dump
from sklearn.dummy import DummyClassifier

from api.main import create_app, create_configured_app
from pathsnap.features import FEATURES


def test_prediction_keeps_reference_out_of_predictor_and_response(tmp_path, caplog):
    trip = tmp_path / "001"
    trip.mkdir()
    files = {
        "nodes": "8.0 49.0\n8.1 49.1\n",
        "arcs": "0 1\n",
        "track": "8.0 49.0 0\n",
        "route": "0\n",
    }
    for suffix, content in files.items():
        (trip / f"001.{suffix}").write_text(content)

    def predict(loaded):
        assert loaded.route is None
        return [0]

    def score(directory, arcs):
        assert (directory / "001.route").read_text().strip() == "0"
        assert arcs == [0]
        return {"accuracy": 1.0}

    client = TestClient(create_app(tmp_path, predict=predict, score=score))
    assert client.get("/trips").json() == {"trips": [{"trip_id": "001", "n_points": 1}]}
    assert client.post("/predict/missing").status_code == 404
    with caplog.at_level(logging.INFO, logger="uvicorn.error.pathsnap"):
        response = client.post("/predict/001")
    payload = response.json()
    assert payload.pop("runtime_s") >= 0
    assert payload == {
        "trip_id": "001", "gps": [[49.0, 8.0]], "scores": {"accuracy": 1.0},
        "route": [[[49.0, 8.0], [49.1, 8.1]]],
    }
    assert "POST /predict/001 200" in caplog.text
    assert TestClient(create_app(tmp_path)).post("/predict/001").status_code == 503


def test_api_allows_the_configured_ui_origin(tmp_path):
    origin = "http://192.168.1.22:8080"
    client = TestClient(create_app(tmp_path, ui_origin=origin))
    assert client.get("/trips", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in client.get(
        "/trips", headers={"Origin": "http://192.168.1.23:8080"}
    ).headers


def test_configured_api_loads_artifact_and_scores(tmp_path, monkeypatch):
    trips = tmp_path / "trips"
    trip = trips / "001"
    trip.mkdir(parents=True)
    (trip / "001.nodes").write_text("8.6 49.8\n8.601 49.8\n")
    (trip / "001.arcs").write_text("0 1\n")
    (trip / "001.track").write_text("8.6001 49.8 0\n8.6002 49.8 1\n")
    (trip / "001.route").write_text("0\n")
    artifact = tmp_path / "model.joblib"
    model = DummyClassifier(strategy="prior").fit([[0] * len(FEATURES)] * 2, [0, 1])
    dump({
        "model": model, "features": FEATURES, "candidates": {"radius_m": 50, "max_k": 8},
        "split_s": 60, "beta_m": 20,
    }, artifact)
    monkeypatch.setenv("PATHSNAP_MODEL_ARTIFACT", str(artifact))
    monkeypatch.setenv("PATHSNAP_EVALUATION_DIR", str(trips))
    with TestClient(create_configured_app()) as client:
        response = client.post("/predict/001")
    assert response.status_code == 200
    assert response.json()["scores"] == {"top1_accuracy": 1.0, "route_continuity": None}
