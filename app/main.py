import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import config, metrics
from app import logging as app_logging
from app.drift import DRIFT_THRESHOLD, DriftTracker
from app.model import ChampionHolder
from app.schemas import PredictRequest, PredictResponse, RefreshResponse
from ml.preprocess import preprocess_features

log = logging.getLogger("app")
holder = ChampionHolder()
drift_tracker = DriftTracker()

# A 3.5-day window never holds all 7 weekdays: this feature would always drift.
DRIFT_IGNORED_FEATURES = {"day_of_week"}


def _on_champion_loaded():
    metrics.model_info.clear()  # one series: the loaded version
    metrics.model_info.labels(str(holder.version)).set(1)
    profile = holder.training_profile() or {}  # no profile -> drift stays off
    drift_tracker.configure(
        {
            name: bins
            for name, bins in profile.get("features", {}).items()
            if name not in DRIFT_IGNORED_FEATURES
        }
    )
    metrics.feature_drift.clear()  # no stale values from the old reference


def _update_drift(version, X):
    drift = drift_tracker.observe(X.iloc[0].to_dict()) or {}
    for name, value in drift.items():
        metrics.feature_drift.labels(name).set(value)
    if drifted := sorted(
        name for name, value in drift.items() if value > DRIFT_THRESHOLD
    ):
        log.warning(
            "data_drift_detected", extra={"model_version": version, "features": drifted}
        )


@asynccontextmanager
async def lifespan(_):
    app_logging.configure_logging()
    try:
        holder.load()
    except Exception:  # start anyway; /health reports 503 until a refresh works
        log.exception("Champion model failed to load at startup")
    else:
        _on_champion_loaded()
    yield


app = FastAPI(title="enercast-mlops", lifespan=lifespan)


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.middleware("http")
async def observe_request(request: Request, call_next):
    # Capped: the id is echoed and logged, a client must not inflate every line.
    request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:12]
    token = app_logging.request_id_var.set(request_id)
    start = time.perf_counter()
    status = 500  # stays 500 if call_next raises, so crashes are still counted
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration = time.perf_counter() - start
        route = request.scope.get("route")  # template, so 404 scans can't mint labels
        path = route.path if route else "unmatched"
        metrics.http_requests_total.labels(request.method, path, str(status)).inc()
        metrics.http_request_duration_seconds.labels(path).observe(duration)
        log.log(
            logging.ERROR if status >= 500 else logging.INFO,
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(duration * 1000, 3),
            },
        )
        app_logging.request_id_var.reset(token)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    if holder.model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "detail": "Champion model failed to load from MLflow Registry.",
            },
        )
    return {"status": "ready", "model_loaded": True, "model_version": holder.version}


@app.get("/model-info")
def model_info():
    if holder.info is None:
        raise HTTPException(503, "Champion model not loaded")
    return holder.info


@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    state = holder.state  # one read: model and info stay consistent across a refresh
    if state is None:
        raise HTTPException(503, "Champion model not loaded")
    model, info = state
    X = preprocess_features(pd.DataFrame([req.model_dump()]))
    y = float(model.predict(X)[0])
    version = str(info["resolved_version"])
    log.info(
        "prediction_made", extra={"model_version": version, "prediction": round(y, 2)}
    )
    try:  # monitoring must never fail a prediction
        metrics.predictions_total.labels(version).inc()
        metrics.prediction_value_wh.labels(version).observe(y)
        _update_drift(version, X)
    except Exception:
        log.exception("monitoring_update_failed")
    return PredictResponse(
        predicted_appliances_wh=round(y, 2),
        model_name=info["model_name"],
        model_alias=info["alias"],
        model_version=info["resolved_version"],
    )


@app.post("/refresh", response_model=RefreshResponse)
def refresh(x_refresh_token: str = Header(default="")):
    # Empty configured token disables the endpoint.
    if not config.REFRESH_API_TOKEN or not secrets.compare_digest(
        x_refresh_token.encode(), config.REFRESH_API_TOKEN.encode()
    ):
        raise HTTPException(401, "Invalid refresh token")
    previous = holder.version
    try:
        holder.load()
    except Exception as exc:
        log.exception("Refresh failed; keeping previous model")
        raise HTTPException(502, "Refresh failed; see server logs") from exc

    _on_champion_loaded()
    return RefreshResponse(
        status="refreshed", previous_version=previous, current_version=holder.version
    )
