import io
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import config, main, model
from app.drift import DriftTracker
from app.logging import JsonFormatter

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
    # No real MLflow download; a fresh drift state per test.
    monkeypatch.setattr(main.holder, "training_profile", lambda: None)
    monkeypatch.setattr(main, "drift_tracker", DriftTracker())

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
    h = model.ChampionHolder()
    h.load()
    assert h.model == "new" and h.version == "3"
    assert h.info["dataset_version"] == "v2"


def test_refresh_non_ascii_token_is_401(client, monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "s3cret")
    with client() as c:
        r = c.post("/refresh", headers={"X-Refresh-Token": "é".encode("latin-1")})
    assert r.status_code == 401


@pytest.fixture
def json_lines():
    """Parsed JSON log lines of the app logger: call after the requests."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logging.getLogger("app").addHandler(handler)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    logging.getLogger("app").removeHandler(handler)


def test_metrics_endpoint_serves_prometheus_text(client):
    with client() as c:
        c.post("/predict", json=PAYLOAD)
        r = c.get("/metrics")
    assert "text/plain" in r.headers["content-type"]
    assert b'predictions_total{model_version="2"}' in r.content
    assert REGISTRY.get_sample_value("model_info", {"model_version": "2"}) == 1.0


def test_request_id_is_echoed_capped_and_on_every_log_line(client, json_lines):
    with client() as c:
        c.post("/predict", json=PAYLOAD, headers={"X-Request-ID": "rid-42"})
        long_id = c.get("/health", headers={"X-Request-ID": "x" * 500})
        generated = c.get("/health")

    lines = [x for x in json_lines() if x.get("request_id") == "rid-42"]
    assert {x["event"] for x in lines} == {"prediction_made", "http_request"}
    assert long_id.headers["X-Request-ID"] == "x" * 64
    assert generated.headers["X-Request-ID"]


def test_unknown_paths_share_one_label_and_crashes_count_as_500(client, monkeypatch):
    def boom(_df):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "preprocess_features", boom)
    crash = {"method": "POST", "path": "/predict", "status": "500"}
    before = REGISTRY.get_sample_value("http_requests_total", crash) or 0

    with client() as c:
        c.get("/no-such-path-123")
        with pytest.raises(RuntimeError):
            c.post("/predict", json=PAYLOAD)

    unmatched = {"method": "GET", "path": "unmatched", "status": "404"}
    assert REGISTRY.get_sample_value("http_requests_total", unmatched)
    assert REGISTRY.get_sample_value("http_requests_total", crash) == before + 1


def test_drift_gauge_and_log_once_the_window_is_full(client, monkeypatch, json_lines):
    bins = {"bin_edges": [10.0], "bin_shares": [0.5, 0.5]}
    profile = {"features": {"lights": bins, "day_of_week": bins}}
    monkeypatch.setattr(main.holder, "training_profile", lambda: profile)
    monkeypatch.setattr(main, "drift_tracker", DriftTracker(window_size=100))

    with client() as c:
        for _ in range(100):
            c.post("/predict", json=PAYLOAD)

    # lights=20: every sample lands in the upper bin
    assert REGISTRY.get_sample_value("feature_drift", {"feature": "lights"}) > 4
    assert (
        REGISTRY.get_sample_value("feature_drift", {"feature": "day_of_week"}) is None
    )
    drift_logs = [x for x in json_lines() if x["event"] == "data_drift_detected"]
    assert [x["features"] for x in drift_logs] == [["lights"]]


def test_refresh_reloads_the_drift_profile_and_clears_stale_gauges(client, monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "tok")
    calls = []
    monkeypatch.setattr(main.holder, "training_profile", lambda: calls.append(1))

    with client() as c:
        c.post("/predict", json=PAYLOAD)
        main.metrics.feature_drift.labels("lights").set(1.5)
        c.post("/refresh", headers={"X-Refresh-Token": "tok"})

    assert len(calls) == 2  # startup and refresh, not per request
    assert REGISTRY.get_sample_value("feature_drift", {"feature": "lights"}) is None


def test_predict_survives_a_monitoring_failure(client, monkeypatch):
    monkeypatch.setattr(main, "_update_drift", lambda *_: 1 / 0)
    with client() as c:
        assert c.post("/predict", json=PAYLOAD).status_code == 200


def test_training_profile_reads_the_loaded_run_or_returns_none(monkeypatch, tmp_path):
    (tmp_path / "p.json").write_text('{"features": {}}')

    def download(run_id, artifact_path):
        if run_id == "gone":
            raise RuntimeError("no artifact")
        assert run_id == "run-7"
        return str(tmp_path / "p.json")

    monkeypatch.setattr(model.mlflow.artifacts, "download_artifacts", download)
    h = model.ChampionHolder()
    h.state = (object(), {"mlflow_run_id": "run-7"})
    assert h.training_profile() == {"features": {}}
    h.state = (object(), {"mlflow_run_id": "gone"})
    assert h.training_profile() is None
