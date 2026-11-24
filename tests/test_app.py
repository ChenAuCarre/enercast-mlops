from fastapi.testclient import TestClient

from app import config, main

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


def client(loaded=True):
    main.holder.model = FakeModel() if loaded else None
    main.holder.info = (
        {"model_name": "m", "alias": "champion", "resolved_version": "2"}
        if loaded
        else None
    )
    main.holder.load = lambda: None  # skip MLflow in lifespan
    return TestClient(main.app)


def test_predict_ok():
    with client() as c:
        r = c.post("/predict", json=PAYLOAD)
    assert r.status_code == 200 and r.json()["predicted_appliances_wh"] == 114.62


def test_predict_out_of_range_and_bad_date():
    with client() as c:
        assert c.post("/predict", json={**PAYLOAD, "RH_1": 101}).status_code == 422
        assert c.post("/predict", json={**PAYLOAD, "date": "nope"}).status_code == 422


def test_health_503_without_model():
    with client(loaded=False) as c:
        assert c.get("/health").status_code == 503


def test_refresh_auth(monkeypatch):
    monkeypatch.setattr(config, "REFRESH_API_TOKEN", "s3cret")
    with client() as c:
        assert c.post("/refresh").status_code == 401
        assert (
            c.post("/refresh", headers={"X-Refresh-Token": "s3cret"}).status_code == 200
        )
