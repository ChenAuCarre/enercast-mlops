import logging
import math
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST

from app import config, metrics
from app import logging as app_logging
from app.drift import DriftTracker
from app.model import ChampionHolder
from app.schemas import PredictRequest, PredictResponse, RefreshResponse
from ml.preprocess import preprocess_features

log = app_logging.get_logger("app")
holder = ChampionHolder()
drift_tracker = DriftTracker()


def _configure_drift(reference):
    drift_tracker.configure(reference)
    metrics.clear_feature_drift()  # no stale gauge values from the old reference


def _load_drift_profile():
    # Network fetch that can be slow or retry: never run it on the startup path.
    version = holder.version
    if version is None:
        return  # nothing loaded: no champion to compare against
    profile = holder.training_profile()
    if profile and profile.get("schema_version") != 2:
        profile = None  # older profiles have no bin shares: PSI would be wrong
    if holder.version != version:
        return  # a refresh moved the champion meanwhile; its own load will configure
    # the artifact nests per-feature stats under "features"
    _configure_drift((profile or {}).get("features"))
    log.log(
        logging.INFO,
        "drift_profile_loaded" if profile else "drift_profile_missing",
        extra={"model_version": str(holder.version) if holder.version else None},
    )


def _load_drift_profile_async():
    threading.Thread(target=_load_drift_profile, daemon=True).start()


@asynccontextmanager
async def lifespan(_):
    app_logging.configure_logging()
    try:
        holder.load()
        metrics.set_model_info(holder.version, loaded=True)
        log.log(
            logging.INFO,
            "model_loaded",
            extra={"model_version": str(holder.version)},
        )
    except Exception:  # start anyway; /health reports 503 until a refresh works
        # Surface the failure through a structured event so that the container
        # logs (and the CI smoke test `docker logs` dump) pinpoint the cause
        # without parsing a free-form traceback message.
        log.log(
            logging.ERROR,
            "champion_model_load_failed",
            extra={"model_version": str(holder.version) if holder.version else None},
            exc_info=True,
        )

    _load_drift_profile_async()

    yield


app = FastAPI(title="enercast-mlops", lifespan=lifespan)


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = app_logging.set_request_id(request_id)
    start = time.perf_counter()
    status = 500  # stays 500 if call_next raises, so crashes are still counted
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_seconds = time.perf_counter() - start
        log.log(
            logging.ERROR
            if status >= 500
            else logging.WARNING
            if status >= 400
            else logging.INFO,
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(duration_seconds * 1000, 3),
            },
        )
        route = request.scope.get("route")  # template, so 404 scans can't mint labels
        metrics.observe_http_request(
            method=request.method,
            path=route.path if route else "unmatched",
            status=status,
            duration_seconds=duration_seconds,
        )
        app_logging.reset_request_id(token)


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
    return Response(
        content=metrics.render_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    state = holder.state  # one read: model and info stay consistent across a refresh
    if state is None:
        raise HTTPException(503, "Champion model not loaded")
    model, info = state
    start = time.perf_counter()
    X = preprocess_features(pd.DataFrame([req.model_dump()]))
    y = float(model.predict(X)[0])
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    try:  # monitoring must never fail a prediction
        metrics.observe_prediction(str(info["resolved_version"]), y)
        drift_tracker.observe({name: float(X.iloc[0][name]) for name in X.columns})
        for name, value in drift_tracker.current().items():
            metrics.set_feature_drift(name, value)
    except Exception:
        log.exception("monitoring_update_failed")

    log.log(
        logging.INFO,
        "prediction_made",
        extra={
            "request_id": app_logging.get_request_id(),
            "model_version": str(info["resolved_version"]),
            "prediction": round(y, 2),
            "latency_ms": latency_ms,
            "feature_count": int(X.shape[1]),
            "missing_count": _missing_count(req.model_dump()),
        },
    )

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

    metrics.set_model_info(holder.version, loaded=True)
    _configure_drift(None)  # old reference no longer matches the new model
    _load_drift_profile_async()

    log.log(
        logging.INFO,
        "model_refreshed",
        extra={
            "request_id": app_logging.get_request_id(),
            "from_version": str(previous) if previous is not None else None,
            "to_version": str(holder.version) if holder.version is not None else None,
        },
    )

    return RefreshResponse(
        status="refreshed", previous_version=previous, current_version=holder.version
    )


def _missing_count(payload: dict) -> int:
    """Count missing values (None or non-finite numbers) in a prediction payload."""
    missing = 0
    for value in payload.values():
        if (
            value is None
            or isinstance(value, float)
            and (math.isnan(value) or math.isinf(value))
        ):
            missing += 1
    return missing
