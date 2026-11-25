# 12-Factor App Assessment (`enercast-mlops`)

This document evaluates `enercast-mlops` against [The Twelve-Factor App](https://12factor.net/) methodology, highlighting what is already working well and where the architecture still needs improvement—especially for multi-worker or multi-instance production scaling.

## Summary

| Factor | Status | Key Strength | Primary Gap |
| :--- | :---: | :--- | :--- |
| **1. Codebase** | Strong | Single repo, tree-hash image promotion | Ephemeral CI staging instead of persistent staging deploy |
| **2. Dependencies** | Strong | `pyproject.toml` + `uv.lock` + slim runtime image | Unpinned base image & `uv` tags in `Dockerfile` |
| **3. Config** | Good | Env vars for secrets/URLs, `gitleaks` in CI | Hardcoded model/drift constants, no startup env validation |
| **4. Backing Services** | Good | MLflow & DVC attached via env vars | No local MLflow in `docker-compose.yml`, no shared state store |
| **5. Build, Release, Run** | Strong | Strict CI/CD stages + decoupled model releases | `POST /refresh` mutates running process state |
| **6. Processes** | Needs Work | No local disk state for inference | In-memory `DriftMonitor` and `ChampionHolder` break multi-replica scaling |
| **7. Port Binding** | Strong | Self-contained `uvicorn` bound to `${PORT:-8000}` | Missing explicit `--proxy-headers` handling behind Render |
| **8. Concurrency** | Needs Work | Thread-safe single-process async/threadpool | Cannot scale to `--workers > 1` or >1 replica safely |
| **9. Disposability** | Good | `exec uvicorn`, graceful startup degradation (`503`) | Synchronous MLflow fetch on cold start, no `HEALTHCHECK` |
| **10. Dev/Prod Parity** | Good | Identical `Dockerfile`, `uv.lock`, and `smoke_test.sh` | No hosted Prometheus/Grafana in prod, no Render staging env |
| **11. Logs** | Good | Unified JSON logs to `stdout` with `X-Request-ID` | No external log aggregation sink configured for prod |
| **12. Admin Processes** | Strong | Training, rollback, and replay run as separate jobs | Admin tasks run in GHA/local venvs rather than a shared image |

---

## Detailed Evaluation

### 1. Codebase
*One codebase tracked in revision control, many deploys.*

* **What went well:**
  * The entire service (`app/`), ML pipeline (`ml/`), monitoring config (`monitoring/`), and CI/CD workflows (`.github/workflows/`) live in a single Git repository.
  * Code promotes sequentially across `dev` → `staging` → `main`, and `cd-staging.yml` tags container images by Git tree hash (`:tree-<tree-hash>`) so `cd-prod.yml` deploys the exact verified image digest to Render without rebuilding.
* **What still needs improvement:**
  * `cd-staging.yml` runs `scripts/smoke_test.sh` against an ephemeral container in GitHub Actions rather than deploying to a persistent staging environment on Render.

### 2. Dependencies
*Explicitly declare and isolate dependencies.*

* **What went well:**
  * Dependencies are declared in `pyproject.toml` with strict separation between runtime dependencies, `dev` tools, and `experiment` tools (`dvc`, `jupyter`, `matplotlib`, etc.).
  * `uv.lock` pins exact versions, `ci-pr.yml` enforces `uv lock --check`, and `Dockerfile` installs only runtime packages into an isolated `/app/.venv` via `uv sync --frozen --no-dev --no-group experiment --no-install-project`.
* **What still needs improvement:**
  * `Dockerfile` uses floating tags (`FROM python:3.12-slim` and `COPY --from=ghcr.io/astral-sh/uv:latest`), which can introduce non-reproducible base OS or `uv` updates across builds. Pinning by digest or exact version would make builds fully deterministic.

### 3. Config
*Store config in the environment.*

* **What went well:**
  * Environment-specific values and credentials (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `REFRESH_API_TOKEN`, `LOG_LEVEL`, `PORT`) are injected via environment variables (`app/config.py`, `mlflow`, `uvicorn`).
  * `.env` and `.dvc/config.local` are git-ignored, and `ci-pr.yml` runs `gitleaks` on every PR.
* **What still needs improvement:**
  * Several operational parameters are hardcoded in Python modules instead of environment variables—for example, `MODEL_NAME` and `CHAMPION_ALIAS` in `ml/registry.py`, and `WINDOW = 500`, `RECOMPUTE_EVERY = 100`, `THRESHOLD = 0.20` in `app/drift.py`.
  * `app/config.py` reads `os.getenv(...)` directly without startup schema validation (e.g., Pydantic `BaseSettings`) to fail fast with clear error messages when required variables are missing or malformed.

### 4. Backing Services
*Treat backing services as attached resources.*

* **What went well:**
  * DagsHub MLflow (tracking + model registry) and DagsHub DVC storage are treated as external attached resources configured via URL and credentials. Switching to a local `file:./mlruns` store or a different MLflow host requires only changing environment variables.
* **What still needs improvement:**
  * `docker-compose.yml` includes Prometheus and Grafana, but does not include a local MLflow server—local container runs still depend on external DagsHub connectivity and credentials in `.env`.
  * There is no shared backing store (e.g., Redis or Postgres) for runtime telemetry/drift state, and MLflow HTTP calls in `app/model.py` / `app/drift.py` rely on default client timeouts.

### 5. Build, Release, Run
*Strictly separate build and run stages.*

* **What went well:**
  * **Build** (`cd-staging.yml`), **Release** (`cd-prod.yml` combining the immutable GHCR digest with Render environment config), and **Run** (`uvicorn` on Render) are cleanly separated.
  * Model releases (`ct-retrain.yml`, `rollback.yml`) are decoupled from code builds via MLflow `@champion` and `POST /refresh`.
* **What still needs improvement:**
  * Calling `POST /refresh` mutates the in-memory model state of a running container during the **Run** stage rather than creating a new immutable release. While intentional for zero-redeploy model swaps, it creates consistency risks if the service scales beyond one process (see Factors 6 and 8).

### 6. Processes
*Execute the app as one or more stateless processes.*

* **What went well:**
  * The container runs as non-root `appuser` with no local disk persistence required for serving predictions; models and reference distributions are loaded from MLflow into memory (`ChampionHolder`).
* **What still needs improvement:**
  * Two in-memory components make the process stateful and prevent safe horizontal scaling:
    1. **`DriftMonitor` (`app/drift.py`)**: Stores the sliding window (`deque(maxlen=500)`) and prediction counter in process memory. Across multiple replicas, each instance computes PSI on a partial sample, and container restarts wipe the drift history.
    2. **`ChampionHolder` (`app/model.py`)**: `POST /refresh` only reloads `@champion` in the single instance/process that handles the request. Behind a load balancer with multiple replicas, other instances would continue serving the previous model version.

### 7. Port Binding
*Export services via port binding.*

* **What went well:**
  * The app is completely self-contained and exports HTTP directly via `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`, binding to the platform-injected `PORT`.
* **What still needs improvement:**
  * `uvicorn` is launched without `--proxy-headers`, which would be needed if downstream logging, redirects, or rate-limiting ever need the client's real IP behind Render's reverse proxy.

### 8. Concurrency
*Scale out via the process model.*

* **What went well:**
  * FastAPI/Uvicorn handles concurrent requests cleanly within a single process: `ChampionHolder` swaps `(model, info)` atomically as a tuple, and `DriftMonitor` guards its window with a `threading.Lock`.
* **What still needs improvement:**
  * The service cannot safely scale out to multiple Uvicorn workers (`--workers > 1`) or multiple Render instances because:
    * `POST /refresh` only updates one worker/instance.
    * `prometheus_client` metrics (`app/metrics.py`) are in-process without `PROMETHEUS_MULTIPROC_DIR` aggregation.
    * `DriftMonitor` state is per-process.
  * Moving drift state to a shared store (or externalizing drift computation to a batch/streaming job) and either polling `@champion` / using event-driven restarts or rolling restarts on model promotion would unlock true horizontal scaling.

### 9. Disposability
*Maximize robustness with fast startup and graceful shutdown.*

* **What went well:**
  * `UV_COMPILE_BYTECODE=1` speeds up Python startup, and `Dockerfile` uses `exec uvicorn ...` so `SIGTERM` is delivered directly to Uvicorn for graceful shutdown.
  * The FastAPI `lifespan` handler catches startup MLflow failures so the container stays up and reports `503` on `/health` rather than entering a crash loop.
* **What still needs improvement:**
  * Cold starts perform synchronous network calls to DagsHub MLflow (`get_champion`, model artifact download, `get_run`, and `training_distribution.json`), which can delay readiness or cause transient `503`s if the registry is slow.
  * `Dockerfile` does not define a `HEALTHCHECK` instruction (health is currently verified externally by Render and `scripts/smoke_test.sh`).

### 10. Dev/Prod Parity
*Keep development, staging, and production as similar as possible.*

* **What went well:**
  * Local Docker runs (`docker compose up --build`), CI (`ci-pr.yml`, `cd-staging.yml`), and production (Render) all use the exact same `Dockerfile`, `uv.lock`, and `scripts/smoke_test.sh`.
* **What still needs improvement:**
  * **Monitoring gap:** Prometheus and Grafana only run in local `docker-compose.yml`; production on Render exposes `/metrics`, but has no hosted Prometheus scraper or Grafana alerting configured.
  * **Staging gap:** Staging validation runs inside GitHub Actions rather than a persistent Render staging service matching production's runtime environment.
  * **Test parity:** `pytest` mocks MLflow and uses fake models, so real registry integration is only exercised in `cd-staging.yml` and `ct-retrain.yml`.

### 11. Logs
*Treat logs as event streams.*

* **What went well:**
  * `app/logs.py` routes application logs as well as `uvicorn`, `uvicorn.error`, `uvicorn.access`, and `mlflow` loggers to `stdout` using a structured `JsonFormatter` with UTC timestamps, log levels, and `X-Request-ID` correlation.
* **What still needs improvement:**
  * Production relies on Render's ephemeral log stream with no external log aggregation/retention sink configured (e.g., Datadog, Loki, or Better Stack).
  * `POST /predict` does not emit structured prediction audit events (e.g., model version and latency) beyond aggregate Prometheus counters.

### 12. Admin Processes
*Run admin/management tasks as one-off processes.*

* **What went well:**
  * Training (`python -m ml.pipeline` in `ct-retrain.yml`), rollbacks (`rollback.yml`), and traffic replay (`scripts/replay.py`) run as isolated one-off processes/workflows using the same repository and `uv.lock`, never as background loops inside the inference API.
* **What still needs improvement:**
  * `ct-retrain.yml` and `rollback.yml` provision their environment via `uv sync` on GitHub Actions runners rather than executing inside a versioned, pre-built training/admin container image.
  * Continuous training is currently `workflow_dispatch`-only rather than scheduled or automatically triggered by drift alerts.
