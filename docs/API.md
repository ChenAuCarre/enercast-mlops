# API Reference (`enercast-mlops`)

The inference service exposes a RESTful JSON API built with FastAPI. Upon startup (or when `/refresh` is invoked), the service queries the MLflow Model Registry, resolves the `@champion` alias to a concrete version number, and loads the corresponding model artifact into memory. If the load fails at startup the service still starts: `/health` returns `503` and `/model-info` / `/predict` return `503` until a `/refresh` succeeds.

* **Base URL**: `https://<your-deployed-service-url>`
* **OpenAPI / Swagger UI**: `https://<your-deployed-service-url>/docs`

---

## Endpoints Overview

| Method | Path | Auth | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Public | Readiness probe; reports `200` only when the `@champion` model is loaded and operational. |
| `GET` | `/model-info` | Public | Returns the resolved MLflow model name, alias, concrete version, and run metadata. |
| `POST` | `/predict` | Public | Predicts appliance energy consumption (`Wh`) for a single 10-minute observation. |
| `POST` | `/refresh` | `X-Refresh-Token` header | Reloads the active `@champion` model from MLflow without container restart. |

---

## 1. Readiness Probe: `GET /health`

Used by container orchestrators and CI/CD smoke tests to verify that the service has successfully downloaded and initialized the approved `@champion` model.

### Responses

* **`200 OK` (Ready)**
  ```json
  {
    "status": "ready",
    "model_loaded": true,
    "model_version": "2"
  }
  ```

* **`503 Service Unavailable` (Not Ready / Model Load Failure)**
  ```json
  {
    "status": "unavailable",
    "model_loaded": false,
    "detail": "Champion model failed to load from MLflow Registry."
  }
  ```

---

## 2. Active Model Metadata: `GET /model-info`

Exposes full traceability of the currently loaded production model.

### Response (`200 OK`)

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

Returns `503` (`{"detail": "Champion model not loaded"}`) if no model is loaded.

---

## 3. Energy Consumption Inference: `POST /predict`

Predicts total household appliance energy consumption (`Appliances`, in Watt-hours `Wh`) from timestamp, lighting usage, indoor multi-zone temperatures/humidity, and local weather station readings.

### Request Schema & Physical Validation Rules

All fields are required. Requests violating physical sensor boundaries or formatting rules are rejected with `422 Unprocessable Entity`.

| Field | Type | Unit | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `string` | ISO / Timestamp | Valid `YYYY-MM-DD HH:MM:SS` | Observation timestamp (used to derive `hour` and `day_of_week`). |
| `lights` | `float` | `Wh` | `>= 0` | Energy use of light fixtures in the house. |
| `T1` – `T9` | `float` | `°C` | `-20.0 <= T <= 60.0` | Indoor temperatures across 9 house zones (kitchen, living, laundry, office, etc.). |
| `RH_1` – `RH_9` | `float` | `%` | `0.0 <= RH <= 100.0` | Indoor relative humidity across 9 house zones. |
| `To` | `float` | `°C` | `-40.0 <= To <= 60.0` | Outdoor temperature from Chievres weather station. |
| `Pressure` | `float` | `mm Hg` | `650.0 <= P <= 850.0` | Atmospheric pressure. |
| `RH_out` | `float` | `%` | `0.0 <= RH_out <= 100.0` | Outdoor relative humidity. |
| `Windspeed` | `float` | `m/s` | `>= 0.0` | Wind speed. |
| `Visibility` | `float` | `km` | `>= 0.0` | Horizontal visibility. |
| `Tdewpoint` | `float` | `°C` | `-50.0 <= Td <= 50.0` | Dew point temperature. |

*(Note: Random noise variables `rv1` and `rv2` from the raw UCI dataset are excluded from the production inference schema.)*

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

* **`503 Service Unavailable`**: no model loaded (`{"detail": "Champion model not loaded"}`).
* **`422 Unprocessable Entity`**: inputs are malformed or physically out-of-range (e.g., `lights < 0` or `RH_1 > 100`):

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

  A bad `date` fails with `type: "value_error"`, `loc: ["body", "date"]` and a message such as `Value error, time data 'x' does not match format '%Y-%m-%d %H:%M:%S'`.

---

## 4. Reload Champion Model: `POST /refresh`

Triggered automatically by `ct-retrain.yml` or `rollback.yml` after the `@champion` alias is updated in MLflow.

### Headers

* `X-Refresh-Token: <REFRESH_API_TOKEN>`

> If `REFRESH_API_TOKEN` is unset, `/refresh` is disabled (always `401`) but the rest of the API still works. If the champion model fails to load at startup, `/health` returns `503` until a successful `/refresh` (needs the token) or a restart.

### Response (`200 OK`)

```json
{
  "status": "refreshed",
  "previous_version": "1",
  "current_version": "2"
}
```

### Error Responses

* **`401 Unauthorized`**: missing/wrong token, or `REFRESH_API_TOKEN` unset (`{"detail": "Invalid refresh token"}`).
* **`502 Bad Gateway`**: the reload failed; the previously loaded model keeps serving (`{"detail": "Refresh failed; see server logs"}`).
