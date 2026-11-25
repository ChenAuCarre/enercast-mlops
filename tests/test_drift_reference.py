import json

from app import main, model
from app.drift import DriftTracker


def _holder(run_id="run-1"):
    h = model.ChampionHolder()
    h.state = (object(), {"resolved_version": "2", "mlflow_run_id": run_id})
    return h


def test_training_profile_uses_the_loaded_run(monkeypatch, tmp_path):
    path = tmp_path / "training-profile.json"
    path.write_text(json.dumps({"features": {}}))
    calls = []

    def download(run_id, artifact_path):
        calls.append(run_id)
        return str(path)

    monkeypatch.setattr(model.mlflow.artifacts, "download_artifacts", download)

    assert _holder("run-7").training_profile() == {"features": {}}
    assert calls == ["run-7"]


def test_training_profile_failure_returns_none(monkeypatch):
    def boom(**_):
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(model.mlflow.artifacts, "download_artifacts", boom)

    assert _holder().training_profile() is None


def _load_reference(monkeypatch, profile):
    tracker = DriftTracker()
    monkeypatch.setattr(main, "drift_tracker", tracker)
    monkeypatch.setattr(main, "_drift_version", None)
    monkeypatch.setattr(main.holder, "state", (object(), {"resolved_version": "2"}))
    monkeypatch.setattr(main.holder, "training_profile", lambda: profile)
    main._load_drift_reference()
    return tracker


def test_old_profile_without_bin_shares_disables_drift(monkeypatch):
    profile = {"schema_version": 1, "features": {"a": {"deciles": [0.0, 1.0]}}}

    assert not _load_reference(monkeypatch, profile).enabled
    assert main._drift_version == "2"  # not retried before the next /refresh


def test_day_of_week_is_not_tracked_for_drift(monkeypatch):
    stats = {"bin_edges": [1.0], "bin_shares": [0.5, 0.5]}
    profile = {"schema_version": 2, "features": {"a": stats, "day_of_week": stats}}

    tracker = _load_reference(monkeypatch, profile)

    assert set(tracker._buffers) == {"a"}
