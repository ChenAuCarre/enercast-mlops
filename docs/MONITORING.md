# Monitoring

The prediction API emits JSON logs, Prometheus metrics and a data drift score;
`docker compose` runs it with Prometheus and Grafana (dashboard and alerts).

## Logs

The `app` logger writes one JSON object per line on stdout, at `LOG_LEVEL` (default `INFO`):

```json
{"timestamp":"2026-11-25T13:11:16.722+00:00","level":"info","event":"prediction_made","logger":"app","model_version":"2","prediction":114.62,"request_id":"5c9be2c025fc"}
```

- `http_request` (method, path, status, duration_ms) per request, level `error` for 5xx.
- `prediction_made` (model_version, prediction) per prediction.
- `data_drift_detected` (warning, the `features` above the threshold).
- `request_id` comes from the `X-Request-ID` header (generated when missing, capped
  at 64 characters) and is echoed in the response.

```bash
docker compose logs api --no-log-prefix | jq -R -c 'fromjson? | select(.level == "error")'
```

## Metrics (`GET /metrics`)

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | `method`, `path` (route template, `unmatched` for unknown paths), `status` |
| `http_request_duration_seconds` | Histogram | `path` |
| `predictions_total` | Counter | `model_version` |
| `prediction_value_wh` | Histogram | `model_version` |
| `model_info` | Gauge | `model_version` of the loaded champion (value 1) |
| `feature_drift` | Gauge | `feature`: PSI against the champion training profile |

## Data drift

- Reference: the `training-profile.json` artifact of the champion's training run
  (per feature: bin edges and training share of each bin), downloaded at startup
  and after `/refresh`. Without it drift is off and the API keeps serving.
- PSI over the last 500 predictions, recomputed every 100 once the window is full.
  Threshold 0.20.
- `day_of_week` is not tracked: a 3.5-day window never holds all 7 weekdays.

## Stack

```bash
docker compose up --build -d
python scripts/replay.py --file data/versions/v2/train.csv --rate 5 --limit 1200
```

- API http://localhost:8000, Prometheus http://localhost:9090/targets,
  Grafana http://localhost:3000 (user `admin`, password `GRAFANA_ADMIN_PASSWORD`, required).
- Ports are bound to `127.0.0.1`.
- `replay.py` shuffles rows: a few days of time-ordered rows read as drift.

Dashboard `WattWise API`: requests per second, 5xx share, p95 latency (without
`/metrics`), predictions per model version, drift per feature, served model version.

Alerts (`monitoring/grafana-alerting.yml`):

1. API down: `up{job="api"}` below 1 (or no data) for 1 minute.
2. Error rate: more than 5% 5xx over 2 minutes. It counts `/health` 503s on purpose:
   with no champion loaded the scrape still succeeds, so this is the alert that fires.
3. Drift: a feature above 0.20 PSI for 2 minutes, only while predictions flow
   (the gauge keeps its last value after traffic stops).
