import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from app import config, drift, logs, main, metrics, model
from ml.drift import training_distribution

PAYLOAD = {
    "date": "2016-05-15 19:30:00",
    "lights": 20,
    "To": 13.8,
    "Pressure": 756.4,
    "RH_out": 65.0,
    "Windspeed": 4.2,
    "Visibility": 40.0,
    "Tdewpoint": 7.3,
    **{f"T{i}": 21.0 for i in range(1, 10)},
    **{f"RH_{i}": 40.0 for i in range(1, 10)},
}


class FakeModel:
    def predict(self, X):
        assert list(X.columns)[-2:] == ["hour", "day_of_week"]
        return [114.617]


@pytest.fixture
def client(monkeypatch):
    """Factory: client(loaded=True) -> TestClient; all patches undone per test."""

    def make(loaded=True):
        info = {"model_name": "m", "alias": "champion", "resolved_version": "2"}
        monkeypatch.setattr(
            main.holder, "state", (FakeModel(), info) if loaded else None
        )
        monkeypatch.setattr(main.holder, "load", lambda: None)  # skip MLflow
        return TestClient(main.app)

    return make


def test_predict_ok(client):
    with client() as c:
        r = c.post("/predict", json=PAYLOAD)
    assert r.status_code == 200 and r.json()["predicted_appliances_wh"] == 114.62


def test_predict_out_of_range_and_bad_date(client):
    with client() as c:
        assert c.post("/predict", json={**PAYLOAD, "RH_1": 101}).status_code == 422
        assert c.post("/predict", json={**PAYLOAD, "date": "nope"}).status_code == 422


def test_health_503_without_model(client):
    with client(loaded=False) as c:
        assert c.get("/health").status_code == 503


def test_refresh_auth(client, monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "s3cret")
    with client() as c:
        assert c.post("/refresh").status_code == 401
        assert (
            c.post("/refresh", headers={"X-Refresh-Token": "s3cret"}).status_code == 200
        )


def test_refresh_failure_keeps_old_model(client, monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "s3cret")
    with client() as c:

        def boom():
            raise RuntimeError("mlflow down")

        monkeypatch.setattr(main.holder, "load", boom)
        r = c.post("/refresh", headers={"X-Refresh-Token": "s3cret"})
        assert r.status_code == 502 and "mlflow down" not in r.text
        assert c.post("/predict", json=PAYLOAD).status_code == 200


def test_root_serves_frontend_ui(client):
    with client() as c:
        r = c.get("/")
        docs = c.get("/docs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "EnerCast MLOps" in r.text
    assert docs.status_code == 200


def test_refresh_disabled_without_token(client, monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "")
    with client() as c:
        assert c.post("/refresh", headers={"X-Refresh-Token": ""}).status_code == 401


def test_model_info_and_503(client):
    with client() as c:
        assert c.get("/model-info").json()["alias"] == "champion"
    with client(loaded=False) as c:
        assert c.get("/model-info").status_code == 503
        assert c.post("/predict", json=PAYLOAD).status_code == 503


def test_holder_load_keeps_old_model_on_failure(monkeypatch):
    h = model.ChampionHolder()
    h.state = ("old", {"resolved_version": "1"})

    def boom():
        raise RuntimeError("no champion")

    monkeypatch.setattr(model.registry, "get_champion", boom)
    with pytest.raises(RuntimeError):
        h.load()
    assert h.model == "old" and h.version == "1"


def test_holder_load_swaps_model(monkeypatch):
    mv = SimpleNamespace(version=3, run_id="r")
    monkeypatch.setattr(model.registry, "get_champion", lambda: mv)
    monkeypatch.setattr(model.registry, "load_model_version", lambda v: "new")
    run = SimpleNamespace(data=SimpleNamespace(params={"dataset_version": "v2"}))
    monkeypatch.setattr(model.mlflow, "get_run", lambda rid: run)
    ref = {"T1": {"edges": [0.0], "proportions": [0.5, 0.5]}}
    monkeypatch.setattr(model.mlflow.artifacts, "load_dict", lambda uri: ref)
    artifact = SimpleNamespace(path="training_distribution.json")
    client = SimpleNamespace(list_artifacts=lambda run_id: [artifact])
    monkeypatch.setattr(model.registry, "get_client", lambda: client)
    monkeypatch.setattr(model.drift.monitor, "reference", None)
    h = model.ChampionHolder()
    h.load()
    assert h.model == "new" and h.version == "3"
    assert model.drift.monitor.reference == ref
    assert h.info["dataset_version"] == "v2"


def test_refresh_non_ascii_token_is_401(client, monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "s3cret")
    with client() as c:
        r = c.post("/refresh", headers={"X-Refresh-Token": "é".encode("latin-1")})
    assert r.status_code == 401


def test_request_id_echoed_or_generated(client):
    with client() as c:
        assert (
            c.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"]
            == "abc"
        )
        assert c.get("/health").headers["X-Request-ID"]


def test_json_log_line_carries_request_id():
    logs.request_id.set("r-1")
    record = logging.LogRecord("app", logging.WARNING, "", 0, "hi %s", ("x",), None)
    line = json.loads(logs.JsonFormatter().format(record))
    assert line["message"] == "hi x" and line["request_id"] == "r-1"
    assert line["level"] == "WARNING"


def test_metrics_exposes_requests_and_predictions(client):
    with client() as c:
        c.post("/predict", json=PAYLOAD)
        body = c.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in body
    assert "http_request_duration_seconds_bucket" in body
    assert 'predictions_total{model_version="2"}' in body
    assert "model_served_version" in body


def test_drift_recomputed_every_100_predictions(caplog):
    m = drift.DriftMonitor()
    m.reference = training_distribution(pd.DataFrame({"T1": np.arange(100.0)}))
    for _ in range(99):
        m.add({"T1": 500.0})
    assert "Drift on T1" not in caplog.text
    m.add({"T1": 500.0})
    assert metrics.DRIFT.labels("T1")._value.get() > drift.THRESHOLD
    assert "Drift on T1" in caplog.text
    assert len(m.window) == 100 and m.window.maxlen == 500


def test_set_reference_clears_stale_drift():
    m = drift.DriftMonitor()
    m.reference = training_distribution(pd.DataFrame({"T1": np.arange(100.0)}))
    for _ in range(100):
        m.add({"T1": 500.0})
    m.set_reference(None)
    assert not m.window and m.count == 0
    assert "feature_drift_psi{" not in generate_latest().decode()


def test_log_level_applies_to_uvicorn_loggers():
    logs.setup("WARNING")
    try:
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
    finally:
        logs.setup("INFO")
