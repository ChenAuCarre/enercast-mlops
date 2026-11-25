# enercast-mlops

**MLOps course project: an end-to-end ML lifecycle for household energy forecasting**

`enercast-mlops` simulates how a machine learning model is trained, released and operated in a production setting. It predicts household appliance energy use (Wh) from the [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) dataset and covers the whole path: versioned data (DVC), tracked experiments and a model registry (MLflow), a containerized inference API, CI/CD with GitHub Actions, and continuous training with gated promotion and rollback.

---

## System Architecture & Tech Stack

```mermaid
flowchart TB
    DVC[("DagsHub Storage<br/>DVC remote")]
    subgraph Training["Continuous Training & Rollback (GitHub Actions)"]
        direction LR
        CT["ct-retrain.yml"]
        RB["rollback.yml"]
    end
    DVC -- "dvc pull<br/>selected version" --> CT
    CT -- "1. log runs; register + move @champion<br/>only if gates pass" --> MLF[("DagsHub MLflow<br/>Tracking + Registry")]
    RB -- "1. move @champion<br/>to an approved version" --> MLF
    subgraph Delivery["CI/CD (GitHub Actions)"]
        direction LR
        STG["cd-staging<br/>build + smoke test"] --> GHCR[("GHCR image<br/>sha / tree tag")] --> PROD["cd-prod<br/>deploy by digest"]
    end
    subgraph Serving["Render"]
        direction LR
        User(["User"]) -- HTTPS --> API["FastAPI + Web UI<br/>(slim Docker)"]
    end
    PROD -- "deploy hook" --> API
    MLF -- "3. load @champion<br/>(also at startup)" --> API
    Training -. "2. only after the alias moved:<br/>POST /refresh, then live check" .-> API
```

* **Application & Inference**: Python, FastAPI, Pydantic, Scikit-Learn
* **Data Versioning**: DVC with DagsHub Storage remote
* **Experiment Tracking & Registry**: MLflow tracking server and model registry on DagsHub
* **Containerization & Operations**: Slim Docker image, configured through environment variables
* **Automation**: GitHub Actions for PR validation, staging verification, production deployment, gated continuous training, and rollback

---

## Live Service

* **Production Web App & API**: `https://enercast-mlops-tree.onrender.com`
* **Interactive Swagger UI**: `https://enercast-mlops-tree.onrender.com/docs`
* **Sample requests, schemas and errors**: [`docs/API.md`](docs/API.md)

Check which model version is serving:

```bash
curl https://enercast-mlops-tree.onrender.com/model-info
```

---

## Documentation

| Doc | Scope |
| :--- | :--- |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Local setup, branch flow and protection, commits, PRs, local checks |
| [`docs/API.md`](docs/API.md) | HTTP contract: endpoints, schemas, validation, errors, examples |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Configuration, workflows and their triggers, deploying, recovery |
| [`ml/README_data.md`](ml/README_data.md) | Dataset source, license, versions, splits, DVC hashes, data contract |
| [`ml/README_experiments.md`](ml/README_experiments.md) | Metric, quality gates, MLflow runs, promotion evidence, CT and rollback logic |

---

## Repository Structure

```text
enercast-mlops/
├── .github/workflows/       # CI/CD, continuous training, rollback
├── app/                     # Inference API and web UI
├── data/                    # DVC-tracked datasets
├── docs/                    # API and deployment docs
├── ml/                      # Training, promotion policy, model registry
├── notebooks/               # Exploratory analysis
├── scripts/                 # Scripts used by GitHub Actions
├── tests/                   # Pytest suite
├── Dockerfile               # Production image
├── pyproject.toml           # Dependencies
└── CONTRIBUTING.md
```
