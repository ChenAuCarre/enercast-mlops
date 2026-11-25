import math
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import config, main, model

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
    # No background profile download (real MLflow) racing the next test's tracker.
    monkeypatch.setattr(main, "_load_drift_profile_async", lambda: None)

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


def test_health_503_when_the_champion_load_fails_at_startup(client, monkeypatch):
    def boom():
        raise RuntimeError("mlflow down")

    events = []

    class RecordingLog:
        def log(self, level, event, extra=None, **kwargs):
            events.append((level, event, extra))

    monkeypatch.setattr(main, "log", RecordingLog())

    c = client(loaded=False)
    monkeypatch.setattr(main.holder, "load", boom)
    with c:
        assert c.get("/health").status_code == 503

    assert any(event == "champion_model_load_failed" for _level, event, _e in events)


def test_health_ready_when_the_drift_profile_is_absent(client, monkeypatch):
    # The champion is loaded but the training profile artifact is missing:
    # the drift monitoring is disabled and /health must still answer 200.
    monkeypatch.setattr(main.holder, "training_profile", lambda: None)

    with client() as c:
        assert c.get("/health").status_code == 200
        assert c.get("/metrics").status_code == 200


def test_metrics_endpoint_returns_the_prometheus_payload(client):
    with client() as c:
        r = c.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert b"http_requests_total" in r.content
    assert b"predictions_total" in r.content


def test_predict_echoes_a_request_id_and_generates_one(client):
    with client() as c:
        echoed = c.post("/predict", json=PAYLOAD, headers={"X-Request-ID": "rid-1"})
        generated = c.post("/predict", json=PAYLOAD)
    assert echoed.status_code == 200
    assert echoed.headers["X-Request-ID"] == "rid-1"
    assert generated.headers["X-Request-ID"]


def test_predict_logs_structured_prediction_and_request_events(client, monkeypatch):
    events = []

    class RecordingLog:
        def log(self, level, event, extra=None, **kwargs):
            events.append((level, event, extra))

    monkeypatch.setattr(main, "log", RecordingLog())

    with client() as c:
        r = c.post("/predict", json=PAYLOAD)
    assert r.status_code == 200

    event_names = [event for _level, event, _extra in events]
    assert "http_request" in event_names
    assert "prediction_made" in event_names

    _level, _event, extra = next(e for e in events if e[1] == "prediction_made")
    assert extra["model_version"] == "2"
    assert "prediction" in extra
    assert "latency_ms" in extra
    assert "feature_count" in extra
    assert "missing_count" in extra
    assert extra["feature_count"] == 27


def test_predict_updates_the_feature_drift_gauge(client, monkeypatch):
    monkeypatch.setattr(
        main.holder,
        "training_profile",
        lambda: {"features": {"lights": {"deciles": [0.0, 10.0, 20.0], "mean": 10.0}}},
    )

    # Load synchronously: stale background threads from earlier tests race otherwise.
    monkeypatch.setattr(main, "_load_drift_profile_async", main._load_drift_profile)

    with client() as c:
        for _ in range(100):
            assert c.post("/predict", json=PAYLOAD).status_code == 200

    value = REGISTRY.get_sample_value("feature_drift", {"feature": "lights"})
    assert value is not None
    assert math.isfinite(value)


def test_unhandled_exception_is_counted_as_a_500(client, monkeypatch):
    def boom(_df):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "preprocess_features", boom)
    labels = {"method": "POST", "path": "/predict", "status": "500"}
    before = REGISTRY.get_sample_value("http_requests_total", labels) or 0

    with client() as c, pytest.raises(RuntimeError):
        c.post("/predict", json=PAYLOAD)

    assert REGISTRY.get_sample_value("http_requests_total", labels) == before + 1


def test_model_info_gauge_reports_the_loaded_version(client, monkeypatch):
    monkeypatch.setattr(main.holder, "load", lambda: None)

    with client():
        pass

    assert REGISTRY.get_sample_value("model_info", {"model_version": "2"}) == 1.0


def test_metrics_path_label_is_bounded_for_unknown_paths(client):
    with client() as c:
        assert c.get("/no-such-path-123").status_code == 404

    assert REGISTRY.get_sample_value(
        "http_requests_total",
        {"method": "GET", "path": "unmatched", "status": "404"},
    )
    assert (
        REGISTRY.get_sample_value(
            "http_requests_total",
            {"method": "GET", "path": "/no-such-path-123", "status": "404"},
        )
        is None
    )
