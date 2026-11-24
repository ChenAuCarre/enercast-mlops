import logging
import secrets
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from app import config
from app.model import ChampionHolder
from app.schemas import PredictRequest, PredictResponse, RefreshResponse
from ml.preprocess import preprocess_features

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


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


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
    if holder.model is None:
        raise HTTPException(503, "Champion model not loaded")
    X = preprocess_features(pd.DataFrame([req.model_dump()]))
    y = float(holder.model.predict(X)[0])
    return PredictResponse(
        predicted_appliances_wh=round(y, 2),
        model_name=holder.info["model_name"],
        model_alias=holder.info["alias"],
        model_version=holder.version,
    )


@app.post("/refresh", response_model=RefreshResponse)
def refresh(x_refresh_token: str = Header(default="")):
    # Empty configured token disables the endpoint.
    if not config.REFRESH_API_TOKEN or not secrets.compare_digest(
        x_refresh_token, config.REFRESH_API_TOKEN
    ):
        raise HTTPException(401, "Invalid refresh token")
    previous = holder.version
    try:
        holder.load()
    except Exception as exc:
        log.exception("Refresh failed; keeping previous model")
        raise HTTPException(502, f"Refresh failed: {exc}") from exc
    return RefreshResponse(
        status="refreshed", previous_version=previous, current_version=holder.version
    )
