import json

from app import main, model
from app.drift import DriftTracker


def _holder(run_id="run-1"):
    h = model.ChampionHolder()
    h.state = (object(), {"resolved_version": "2", "mlflow_run_id": run_id})
    return h


def test_training_profile_uses_the_loaded_run_and_caches_success(monkeypatch, tmp_path):
    path = tmp_path / "training-profile.json"
    path.write_text(json.dumps({"features": {}}))
    calls = []

    def download(run_id, artifact_path):
        calls.append(run_id)
        return str(path)

    monkeypatch.setattr(model.mlflow.artifacts, "download_artifacts", download)
    h = _holder("run-7")

    assert h.training_profile() == {"features": {}}
    assert h.training_profile() == {"features": {}}
    assert calls == ["run-7"]


def test_training_profile_failure_is_not_cached(monkeypatch):
    def boom(**_):
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(model.mlflow.artifacts, "download_artifacts", boom)
    h = _holder()

    assert h.training_profile() is None
    assert h._training_profiles == {}


def test_stale_profile_load_does_not_overwrite_a_newer_champion(monkeypatch):
    tracker = DriftTracker()
    monkeypatch.setattr(main, "drift_tracker", tracker)

    def slow_profile():
        # a refresh lands while the download is in flight
        main.holder.state = (object(), {"resolved_version": "3"})
        return {"schema_version": 2, "features": {"a": {"mean": 1.0}}}

    monkeypatch.setattr(main.holder, "state", (object(), {"resolved_version": "2"}))
    monkeypatch.setattr(main.holder, "training_profile", slow_profile)

    main._load_drift_profile()

    assert not tracker.enabled


def test_old_profile_without_bin_shares_disables_drift(monkeypatch):
    tracker = DriftTracker()
    monkeypatch.setattr(main, "drift_tracker", tracker)
    monkeypatch.setattr(main.holder, "state", (object(), {"resolved_version": "2"}))
    monkeypatch.setattr(
        main.holder,
        "training_profile",
        lambda: {"schema_version": 1, "features": {"a": {"deciles": [0.0, 1.0]}}},
    )

    main._load_drift_profile()

    assert not tracker.enabled
