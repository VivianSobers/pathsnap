"""FastAPI app: trip listing and prediction endpoints."""

from collections.abc import Callable, Sequence
import logging
import math
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

from api.evaluate import score_trip
from pathsnap.io import Trip, list_trips, load_trip
from pathsnap.predict import predict_trip as decode_trip


logger = logging.getLogger("uvicorn.error").getChild("pathsnap")
Predictor = Callable[[Trip], Sequence[int]]
Scorer = Callable[[Path, list[int]], dict[str, float | None]]


class Prediction(BaseModel):
    trip_id: str
    gps: list[tuple[float, float]]
    route: list[list[tuple[float, float]]]
    scores: dict[str, float | None]
    runtime_s: float


class TripSummary(BaseModel):
    trip_id: str
    n_points: int


def create_app(evaluation_dir: Path | None, *, predict: Predictor | None = None,
               score: Scorer | None = None, ui_origin: str | None = None) -> FastAPI:
    """Serve reserved trips with independently injected prediction and scoring."""
    app = FastAPI(title="PathSnap")
    if ui_origin is not None:
        app.add_middleware(CORSMiddleware, allow_origins=[ui_origin], allow_methods=["GET", "POST"])

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        logger.info("%s %s %d %.1f ms", request.method, request.url.path,
                    response.status_code, (perf_counter() - start) * 1000)
        return response

    def trip_dirs():
        if evaluation_dir is None or not evaluation_dir.is_dir():
            raise HTTPException(503, "Evaluation trip directory is unavailable")
        return {path.name: path for path in list_trips(evaluation_dir) if not path.is_symlink()}

    @app.get("/trips")
    def trips() -> dict[str, list[TripSummary]]:
        summaries = []
        for trip_id, directory in trip_dirs().items():
            track_path = (directory / trip_id).with_suffix(".track")
            with track_path.open(encoding="utf-8") as track:
                summaries.append(TripSummary(trip_id=trip_id, n_points=sum(bool(line.strip()) for line in track)))
        return {"trips": summaries}

    @app.post("/predict/{trip_id}", response_model=Prediction)
    def predict_trip(trip_id: str) -> Prediction:
        start_time = perf_counter()
        trip_dir = trip_dirs().get(trip_id)
        if trip_dir is None:
            raise HTTPException(404, "Unknown trip")
        if predict is None or score is None:
            raise HTTPException(503, "Prediction pipeline is unavailable")

        trip = load_trip(trip_dir, include_route=False)
        arc_ids = [int(arc) for arc in predict(trip)]
        if len(arc_ids) != len(trip.track) or any(arc < -1 or arc >= len(trip.arcs) for arc in arc_ids):
            logger.error("Invalid prediction for trip %s", trip_id)
            raise HTTPException(500, "Invalid prediction")

        segments = []
        for arc in arc_ids:
            if arc == -1:
                continue
            start, end = trip.arcs.iloc[arc]
            coords = trip.nodes.iloc[[start, end]][["lat", "lon"]].to_numpy().tolist()
            segments.append(coords)
        gps = [
            (lat, lon) for lat, lon in trip.track[["lat", "lon"]].itertuples(index=False, name=None)
            if math.isfinite(lat) and math.isfinite(lon)
        ]
        scores = {key: value if value is None or math.isfinite(value) else None
                  for key, value in score(trip_dir, arc_ids).items()}
        return Prediction(trip_id=trip_id, gps=gps, route=segments, scores=scores,
                          runtime_s=perf_counter() - start_time)

    return app


def create_configured_app():
    """Load an artifact exported by this project for the configured evaluation directory."""
    evaluation_path = os.getenv("PATHSNAP_EVALUATION_DIR")
    artifact_path = os.getenv("PATHSNAP_MODEL_ARTIFACT")
    artifact = load(artifact_path) if artifact_path else None
    return create_app(
        Path(evaluation_path) if evaluation_path else None,
        predict=(lambda trip: decode_trip(trip, artifact)) if artifact is not None else None,
        score=score_trip,
        ui_origin=os.getenv("PATHSNAP_UI_ORIGIN"),
    )


app = create_configured_app()
