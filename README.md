# enercast-mlops

**End-to-End Production MLOps Reference Project for Time-Series Energy Forecasting**

`enercast-mlops` is a production-ready machine learning reference implementation built around the [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) dataset. It demonstrates how to take a time-series regression model from versioned data ingestion to automated Continuous Integration, Continuous Deployment, and Continuous Training (**CI/CD/CT**) with zero-downtime model promotion and rollback.

---

## System Architecture & Tech Stack

```mermaid
flowchart TB
    subgraph Training["Continuous Training"]
        direction LR
        DVC[("DagsHub Storage<br/>DVC data v1/v2")] --> CT["GitHub Actions<br/>CT pipeline"]
    end
    CT -- "log + register<br/>(@champion if gates pass)" --> MLF[("DagsHub MLflow<br/>Tracking + Registry")]
    subgraph Serving["Serving"]
        direction LR
        User(["User"]) -- HTTPS --> Host["Cloud Hosting"] --> API["FastAPI<br/>(Slim Docker)"]
    end
    MLF -- "load @champion" --> API
```

* **Application & Inference**: Python, FastAPI, Pydantic, Scikit-Learn
* **Data Versioning**: DVC with DagsHub Storage remote
* **Experiment Tracking & Registry**: MLflow tracking server and model registry on DagsHub
* **Containerization & Operations**: Slim Docker image adhering to the [Twelve-Factor App](docs/DEPLOYMENT.md) methodology
* **Automation**: GitHub Actions for PR validation, staging verification, production deployment, gated continuous training, and rollback

---

## Live Service & Quickstart

* **Production Web App & API**: `https://enercast-mlops-tree.onrender.com`
* **Interactive Swagger UI**: `https://enercast-mlops-tree.onrender.com/docs`
* **API Contract & Schemas**: See **[`docs/API.md`](docs/API.md)** for full request/response specifications, validation constraints, and status codes.

### Quick Sample Request (`POST /predict`)

```bash
curl -X POST "https://enercast-mlops-tree.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2016-05-15 19:30:00",
    "lights": 20,
    "T1": 22.5, "RH_1": 41.2, "T2": 21.8, "RH_2": 40.5,
    "T3": 23.1, "RH_3": 39.8, "T4": 21.4, "RH_4": 38.9,
    "T5": 20.2, "RH_5": 48.1, "T6": 14.3, "RH_6": 62.0,
    "T7": 20.8, "RH_7": 37.5, "T8": 22.0, "RH_8": 42.1,
    "T9": 20.5, "RH_9": 41.0, "To": 13.8, "Pressure": 756.4,
    "RH_out": 65.0, "Windspeed": 4.2, "Visibility": 40.0, "Tdewpoint": 7.3
  }'
```

---

## Automation Pipelines (CI/CD/CT)

### Branch flow

```text
feature/* ──PR──► dev ──PR──► staging ──PR──► main
```

* Short-lived feature branches (`feature/*`, `feat/*`, `fix/*`, `docs/*`, `ci/*`, ...) are created from `dev` and merged into `dev` through a reviewed pull request.
* `dev` is promoted to `staging`, and `staging` to `main`, only through pull requests. `enforce-source-branch.yml` rejects any other source branch.
* Repository rulesets protect `dev`, `staging` and `main`: pull request required, no force-push or deletion, and the `ci-pass` check must succeed (plus `check-source` on `staging` and `main`).

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for naming and commit conventions.

### Workflows

All GitHub Actions workflows live under `.github/workflows/`:

| Workflow | Trigger | Purpose |
| :--- | :--- | :--- |
| **`ci-pr.yml`** | `pull_request` $\rightarrow$ `dev`, `staging`, `main` | Lockfile check, actionlint, secret scan, `ruff` lint/format, `pytest` suite (fake models and mocked registry, no network), Docker build, container smoke test, and PR title check. `ci-pass` aggregates them as the single required check. |
| **`enforce-source-branch.yml`** | `pull_request` $\rightarrow$ `staging`, `main` | Only `dev` may open a PR into `staging` and only `staging` into `main`. |
| **`cd-staging.yml`** | `push` (merge) $\rightarrow$ `staging` | Builds the release candidate, smoke-tests it against the current `@champion`, and publishes the verified image to GHCR (tagged by commit SHA and tree hash). |
| **`cd-prod.yml`** | `push` (merge) $\rightarrow$ `main` | Selects the image verified in staging (fails if none), deploys it by digest to Render, waits for the deploy to go live, and runs a live prediction + model version smoke test. |
| **`ct-retrain.yml`** | `workflow_dispatch` (`dataset-version`: `v1` or `v2`) | Fetches DVC snapshot from DagsHub, validates data contract, evaluates `@champion`, trains & gates candidate, promotes if passing, and refreshes production. |
| **`rollback.yml`** | `workflow_dispatch` (`target-version`, `reason`) | Reassigns `@champion` alias to a previously approved version, refreshes production, and verifies via live prediction. |

---

## Repository Structure

```text
enercast-mlops/
├── .github/workflows/       # ci-pr, enforce-source-branch, cd-staging, cd-prod, ct-retrain, rollback
├── .dvc/                    # DVC remote configuration (DagsHub Storage)
├── app/                     # FastAPI inference service, MLflow champion loader, web UI (static/)
├── data/                    # DVC pointers: raw/, versions/v1, versions/v2, versions/test
├── docs/
│   ├── API.md               # REST API schemas, sample payloads, validation rules & error codes
│   └── DEPLOYMENT.md        # Workflows, deployment reproduction, failure recovery, twelve-factor
├── ml/                      # Data prep, splits, training, promotion policy, registry, CT pipeline
│   ├── README_data.md       # Dataset source, license, versions, splits, DVC hashes
│   └── README_experiments.md # Metric, quality gates, MLflow runs, promotion evidence
├── notebooks/               # Exploratory data analysis
├── scripts/smoke_test.sh    # Readiness, model version and live prediction check
├── tests/                   # Pytest suite: API, preprocessing, splits, training, policy, registry
├── Dockerfile               # Slim production image (runtime dependencies only)
├── .dockerignore            # Keeps data, tests, docs and secrets out of the image
├── pyproject.toml           # Dependencies, split into runtime / dev / experiment groups
├── uv.lock                  # Locked dependency versions
└── CONTRIBUTING.md          # Developer setup, Git branching flow, testing & PR checklist
```
