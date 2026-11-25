import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import config, drift, logs, metrics
from app.model import ChampionHolder
from app.schemas import PredictRequest, PredictResponse, RefreshResponse
from ml.preprocess import preprocess_features

logs.setup(config.LOG_LEVEL)
log = logging.getLogger("app")
holder = ChampionHolder()


@asynccontextmanager
async def lifespan(_):
    try:
        holder.load()
    except Exception:  # start anyway; /health reports 503 until a refresh works
        log.exception("Champion model failed to load at startup")
    yield


app = FastAPI(title="enercast-mlops", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = logs.request_id.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Answer here: the server's outer 500 handler would drop X-Request-ID.
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        response = PlainTextResponse("Internal Server Error", status_code=500)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"  # bounded label cardinality
    metrics.REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    metrics.LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
    logs.request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


STATIC_DIR = Path(__file__).resolve().parent / "static"


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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    state = holder.state  # one read: model and info stay consistent across a refresh
    if state is None:
        raise HTTPException(503, "Champion model not loaded")
    model, info = state
    X = preprocess_features(pd.DataFrame([req.model_dump()]))
    y = float(model.predict(X)[0])
    metrics.PREDICTIONS.labels(info["resolved_version"]).inc()
    drift.monitor.add(X.iloc[0].to_dict())
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
    return RefreshResponse(
        status="refreshed", previous_version=previous, current_version=holder.version
    )
