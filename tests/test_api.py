import logging

from fastapi.testclient import TestClient

from api.main import create_app


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
