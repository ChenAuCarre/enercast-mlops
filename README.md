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

## Live Service

* **Production Web App & API**: `https://enercast-mlops-tree.onrender.com`
* **Interactive Swagger UI**: `https://enercast-mlops-tree.onrender.com/docs`
* **Sample requests, schemas and errors**: [`docs/API.md`](docs/API.md)

---

## Documentation

| Doc | Scope |
| :--- | :--- |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Local setup, branch flow and protection, commits, PRs, local checks |
| [`docs/API.md`](docs/API.md) | HTTP contract: endpoints, schemas, validation, errors, examples |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Configuration, workflows and their triggers, deploying, recovery, twelve-factor |
| [`ml/README_data.md`](ml/README_data.md) | Dataset source, license, versions, splits, DVC hashes, data contract |
| [`ml/README_experiments.md`](ml/README_experiments.md) | Metric, quality gates, MLflow runs, promotion evidence, CT and rollback logic |

---

## Repository Structure

```text
enercast-mlops/
├── .github/workflows/       # ci-pr, enforce-source-branch, cd-staging, cd-prod, ct-retrain, rollback
├── .dvc/                    # DVC remote configuration (DagsHub Storage)
├── app/                     # FastAPI inference service, MLflow champion loader, web UI (static/)
├── data/                    # DVC pointers: raw/, versions/v1, versions/v2, versions/test
├── docs/                    # API.md, DEPLOYMENT.md
├── ml/                      # Data prep, splits, training, promotion policy, registry, CT pipeline, ML docs
├── notebooks/               # Exploratory data analysis
├── scripts/smoke_test.sh    # Readiness, model version and live prediction check
├── tests/                   # Pytest suite: API, preprocessing, splits, training, policy, registry
├── Dockerfile               # Slim production image (runtime dependencies only)
├── .dockerignore            # Keeps data, tests, docs and secrets out of the image
├── pyproject.toml           # Dependencies, split into runtime / dev / experiment groups
├── uv.lock                  # Locked dependency versions
└── CONTRIBUTING.md
```
