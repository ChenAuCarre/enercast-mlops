# Monitoring, Observability and Alerting

This document describes the monitoring of the WattWise prediction API: structured
logs, Prometheus metrics, data drift, and alerting. It adapts the monitoring
exercise to the WattWise architecture (energy forecasting regression instead of
the bank-marketing classification of the exercise).

## Overview

The deployed API exposes three monitoring concerns:

1. **Structured logs** — every line emitted by the app is one JSON object written
   to standard output.
2. **Metrics** — the API serves Prometheus metrics at `GET /metrics`.
3. **Data drift** — live predictions are compared with the champion training
   profile using a Population Stability Index (PSI).

The champion training profile (`training-profile.json`) is logged by the
training run (`ml.train`) from the training features; the API downloads it from
the MLflow run of the champion version on its first prediction (and again after
a `/refresh`).

## Structured logs

- Log level comes from the `LOG_LEVEL` environment variable (default `INFO`).
- Every line is one JSON object, for example:

```json
{"timestamp":"2026-11-25T13:11:16.722+00:00","level":"info","event":"prediction_made","logger":"app","request_id":"5c9be2c025fc","model_version":"2","prediction":114.62,"latency_ms":4.167,"feature_count":27}
```

- Request lifecycle events: `http_request` (method, path, status, duration_ms) and
  `prediction_made` (model_version, prediction, latency_ms, feature_count).
- Model lifecycle events: `model_loaded`, `champion_model_load_failed`,
  `drift_profile_loaded`, `drift_profile_missing`, `model_refreshed`.
- 5xx responses are logged with level `error`, 4xx with `warning`.
- The request id comes from the `X-Request-ID` header (generated when missing) and
  is echoed in the response and attached to every log line of the request.

Inspect logs with Docker:

```bash
docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.level == "error")'
```

## Metrics

The API serves its metrics at `GET /metrics` through `prometheus_client`.

| Metric | Type | Labels / meaning |
|---|---|---|
| `http_requests_total` | Counter | `method`, `path` (route template, `unmatched` for unknown paths), `status` |
| `http_request_duration_seconds` | Histogram | `path` |
| `predictions_total` | Counter | `model_version` |
| `prediction_value_wh` | Histogram | `model_version` (predicted appliance energy in Wh) |
| `model_info` | Gauge | `model_version` of the loaded champion, value 1; no series until a model loads |
| `feature_drift` | Gauge | `feature` (PSI against the champion training profile) |

## Data drift

- Reference: `training-profile.json` artifact of the champion version
  (per feature: bin edges and the real training share of each bin).
- Live distribution: last 500 predictions per feature (`DRIFT_WINDOW_SIZE`).
- Recomputed every 100 predictions (`DRIFT_WINDOW_RESET_EVERY`).
- Threshold: `DRIFT_THRESHOLD = 0.20` (PSI).
- `day_of_week` is not tracked: a 3.5-day window never holds all 7 weekdays.
- When the champion has no training profile, drift monitoring is disabled and a
  `drift_profile_missing` warning is logged; the service keeps
  serving.

## Stack

```bash
docker compose up --build -d
docker compose ps
docker compose logs api --no-log-prefix
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090/targets
- Grafana: http://localhost:3000 (user `admin`, password `GRAFANA_ADMIN_PASSWORD`, default `admin`)

All ports are bound to `127.0.0.1`. Set `ALERT_EMAIL_TO` and the `SMTP_*` variables
for alert emails (see `.env.example`).

## Replay

```bash
python scripts/replay.py --file data/versions/v2/train.csv --url http://localhost:8000/predict --limit 20
python scripts/replay.py --file data/versions/v2/train.csv --rate 5 --limit 1200 --broken
python scripts/replay.py --file data/versions/v2/train.csv --ordered --limit 20
python scripts/replay.py --file data/versions/v2/train.csv --shift To=10 --limit 600
```

Rows are shuffled by default: about 3.5 days of time-ordered rows look different from
months of training data and trip the drift alert. `--ordered` keeps file order.

`--broken` sends rows with a missing feature column (expected `422` responses).
It exercises the 4xx logs and metrics; it does **not** trigger the 5xx error-rate
alert, which needs real server errors (for example a champion that is not loaded,
answering `503`).

`--shift To=10` adds a delta to a feature in every row, to simulate drift and
trigger the drift alert after 100+ predictions (repeat the flag for several features).

## Alerts

The Grafana alerting rules provisioned in `monitoring/grafana-alerting.yml`:

1. API down: the Prometheus scrape of the API (`up{job="api"}`) has failed for 1 minute (no data counts as down).
2. High error rate: more than 5% of all API requests (including `/health` checks and
   Prometheus scrapes) failed with a 5xx status
   over 2 minutes.
3. Data drift: a feature drifted above 0.20 PSI for 2 minutes.

The no-data state is set to `OK` (shown as Normal) for the error-rate and drift rules so that
restarts of the API do not fire false alerts.

## Dashboard

The dashboard `WattWise API` (provisioned JSON) contains:

1. Requests per second.
2. Share of requests that failed (5xx), in %.
3. 95th percentile latency.
4. Predictions per model version.
5. Drift per feature, with a threshold line at 0.20.
6. The model version currently served.

## Rollback watch

After a rollback (`rollback.yml`) and a `/refresh` or a service restart:

1. Note the champion version before the rollback.
2. Run the replay for the new champion's own training data.
3. Check the dashboard and logs: which model version served each prediction,
   before and after the rollback.

## Related documents

- [`../docs/API.md`](API.md) - endpoint contract.
- [`../docs/DEPLOYMENT.md`](DEPLOYMENT.md) - deployments and workflows.
- [`../ml/README_experiments.md`](../ml/README_experiments.md) - promotion evidence.
