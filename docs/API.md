# API Reference

The `enercast-mlops` inference service is a FastAPI application serving both a RESTful JSON API and a static web UI.

* **Production Base URL**: `https://enercast-mlops-tree.onrender.com`
* **Interactive Swagger UI**: `https://enercast-mlops-tree.onrender.com/docs`

---

## Runtime Model Lifecycle

On startup, the service queries the MLflow Model Registry, resolves `wattwise-energy-forecaster@champion` to a concrete version, and loads that model artifact into memory:

* **Startup failure resilience:** If the initial load fails, the process still starts so orchestrators can inspect `/health`, which returns `503 Service Unavailable`. `/model-info` and `/predict` also return `503` until a `POST /refresh` succeeds or the service restarts with a reachable champion.
* **Zero-downtime reloads:** `POST /refresh` loads the newly resolved `@champion` model and swaps the model and metadata atomically. If a reload fails, the service returns `502 Bad Gateway` and continues serving the previously loaded model.

---

## Endpoints Overview

| Method | Path | Auth | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Public | Serves the interactive browser web UI (excluded from OpenAPI schema). |
| `GET` | `/health` | Public | Readiness probe; returns `200` only when a `@champion` model is loaded. |
| `GET` | `/model-info` | Public | Returns registry metadata for the currently loaded model version. |
| `POST` | `/predict` | Public | Predicts household appliance energy consumption (`Wh`) for one 10-minute observation. |
| `POST` | `/refresh` | `X-Refresh-Token` | Reloads `@champion` from MLflow without restarting the container. |

---

## 1. Readiness Probe: `GET /health`

Used by Render, Docker health checks, and CI/CD smoke tests to confirm that the service has initialized a champion model.

### Responses

* **`200 OK` (Ready)**

  ```json
  {
    "status": "ready",
    "model_loaded": true,
    "model_version": "2"
  }
  ```

* **`503 Service Unavailable` (Not Ready)**

  ```json
  {
    "status": "unavailable",
    "model_loaded": false,
    "detail": "Champion model failed to load from MLflow Registry."
  }
  ```

---

## 2. Active Model Metadata: `GET /model-info`

Returns traceability metadata for the loaded `@champion` model. `dataset_version` is `null` if the originating MLflow run did not log that parameter.

### Responses

* **`200 OK`**

  ```json
  {
    "model_name": "wattwise-energy-forecaster",
    "alias": "champion",
    "resolved_version": "2",
    "mlflow_run_id": "a1b2c3d4e5f67890",
    "dataset_version": "v2",
    "loaded_at_utc": "2026-11-21T14:30:00+00:00"
  }
  ```

* **`503 Service Unavailable`** — returned when no model is loaded:

  ```json
  {
    "detail": "Champion model not loaded"
  }
  ```

---

## 3. Energy Consumption Inference: `POST /predict`

Predicts total household appliance energy consumption (`Appliances`, in `Wh`, rounded to 2 decimal places) from an observation timestamp, lighting energy use, nine building sensor zones, and outdoor weather readings.

### Request Schema & Validation Rules

All 26 fields are required. Requests with missing fields, invalid timestamps, or out-of-range sensor values are rejected with `422 Unprocessable Entity` (enforced in `app/schemas.py`).

| Field | Type | Unit | Validation Rule | Description |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `string` | Timestamp | Strict `YYYY-MM-DD HH:MM:SS` | Observation timestamp used to derive `hour` and `day_of_week`. ISO-8601 strings with `T` or timezone offsets are rejected. |
| `lights` | `float` | `Wh` | `>= 0.0` | Energy use of household light fixtures. |
| `T1` – `T9` | `float` | `°C` | `-20.0 <= T <= 60.0` | Temperatures across 9 house sensor zones. |
| `RH_1` – `RH_9` | `float` | `%` | `0.0 <= RH <= 100.0` | Relative humidity across the same 9 sensor zones. |
| `To` | `float` | `°C` | `-40.0 <= To <= 60.0` | Outdoor temperature from the Chièvres weather station. |
| `Pressure` | `float` | `mm Hg` | `650.0 <= Pressure <= 850.0` | Atmospheric pressure. |
| `RH_out` | `float` | `%` | `0.0 <= RH_out <= 100.0` | Outdoor relative humidity. |
| `Windspeed` | `float` | `m/s` | `>= 0.0` | Wind speed. |
| `Visibility` | `float` | `km` | `>= 0.0` | Horizontal visibility. |
| `Tdewpoint` | `float` | `°C` | `-50.0 <= Tdewpoint <= 50.0` | Dew point temperature. |

> The random noise columns `rv1` and `rv2` present in the raw UCI dataset are intentionally excluded from the inference schema.

### Example Request

```json
{
  "date": "2016-05-15 19:30:00",
  "lights": 20,
  "T1": 22.5, "RH_1": 41.2,
  "T2": 21.8, "RH_2": 40.5,
  "T3": 23.1, "RH_3": 39.8,
  "T4": 21.4, "RH_4": 38.9,
  "T5": 20.2, "RH_5": 48.1,
  "T6": 14.3, "RH_6": 62.0,
  "T7": 20.8, "RH_7": 37.5,
  "T8": 22.0, "RH_8": 42.1,
  "T9": 20.5, "RH_9": 41.0,
  "To": 13.8, "Pressure": 756.4,
  "RH_out": 65.0, "Windspeed": 4.2,
  "Visibility": 40.0, "Tdewpoint": 7.3
}
```

### Example Response (`200 OK`)

```json
{
  "predicted_appliances_wh": 114.62,
  "model_name": "wattwise-energy-forecaster",
  "model_alias": "champion",
  "model_version": "2"
}
```

### Error Responses

* **`503 Service Unavailable`** — no model is currently loaded:

  ```json
  {
    "detail": "Champion model not loaded"
  }
  ```

* **`422 Unprocessable Entity`** — malformed input or sensor value outside physical bounds (for example, `RH_1 > 100`):

  ```json
  {
    "detail": [
      {
        "type": "less_than_equal",
        "loc": ["body", "RH_1"],
        "msg": "Input should be less than or equal to 100",
        "input": 101,
        "ctx": {"le": 100.0}
      }
    ]
  }
  ```

  An invalid `date` value returns `type: "value_error"` at `loc: ["body", "date"]` with a message such as `Value error, time data 'x' does not match format '%Y-%m-%d %H:%M:%S'`.

---

## 4. Reload Champion Model: `POST /refresh`

Instructs the running service to re-resolve `wattwise-energy-forecaster@champion` in MLflow and load the updated version into memory. Called automatically by `ct-retrain.yml` and `rollback.yml` after the alias moves.

### Authentication Header

* `X-Refresh-Token: <REFRESH_API_TOKEN>`

If the `REFRESH_API_TOKEN` environment variable is unset or empty on the server, `/refresh` is disabled and always returns `401 Unauthorized`; all other endpoints continue operating normally.

### Responses

* **`200 OK`** (`previous_version` is `null` if no model was loaded before the refresh)

  ```json
  {
    "status": "refreshed",
    "previous_version": "1",
    "current_version": "2"
  }
  ```

* **`401 Unauthorized`** — missing or mismatched `X-Refresh-Token`, or `REFRESH_API_TOKEN` is unset:

  ```json
  {
    "detail": "Invalid refresh token"
  }
  ```

* **`502 Bad Gateway`** — reloading from MLflow failed; the previously loaded model remains active:

  ```json
  {
    "detail": "Refresh failed; see server logs"
  }
  ```
