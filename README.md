# enercast-mlops

**MLOps course project: an end-to-end ML lifecycle for household energy forecasting**

`enercast-mlops` simulates how a machine learning model is trained, released and operated in a production setting. It predicts household appliance energy use (Wh) from the [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) dataset and covers the whole path: versioned data (DVC), tracked experiments and a model registry (MLflow), a containerized inference API, CI/CD with GitHub Actions, and continuous training with gated promotion and rollback.

---

## System Architecture & Tech Stack

```mermaid
flowchart TB
    Repo["GitHub repo<br/>code + DVC pointers"]
    DVC[("DagsHub Storage<br/>DVC remote")]
    subgraph GHA["GitHub Actions"]
        direction LR
        CR["Code release<br/>test, build, verify"]
        MR["Model release<br/>train, gate, promote / rollback"]
    end
    GHCR[("GHCR<br/>verified images")]
    MLF[("DagsHub MLflow<br/>Tracking + Registry")]
    subgraph Render["Render"]
        API["FastAPI + Web UI<br/>(slim Docker)"]
    end
    User(["User"])

    Repo -- "PR / merge" --> CR
    Repo -- "manual dispatch" --> MR
    DVC -- "dataset" --> MR
    CR -- "image" --> GHCR
    GHCR -- "deploy by digest" --> API
    MR -- "1. log runs, move @champion" --> MLF
    MR -. "2. after the move:<br/>POST /refresh" .-> API
    MLF -- "3. load @champion<br/>(also at startup)" --> API
    User -- HTTPS --> API
```

* **Application & Inference**: Python, FastAPI, Pydantic, Scikit-Learn
* **Data Versioning**: DVC with DagsHub Storage remote
* **Experiment Tracking & Registry**: MLflow tracking server and model registry on DagsHub
* **Containerization & Operations**: Slim Docker image, configured through environment variables
* **Automation**: GitHub Actions for PR validation, staging verification, production deployment, gated continuous training, and rollback

Code and model are released separately and meet only in the running service:
Render runs the image, and the image loads `@champion` from MLflow at runtime.
A code deploy never changes the model; a model change never redeploys the code.

---

## Branch Flow & Workflows

```mermaid
flowchart LR
    F["feature/*, fix/*,<br/>docs/*, ci/*"] -- "PR: ci-pass" --> DEV["dev"]
    DEV -- "PR: ci-pass, check-source,<br/>1 approval" --> STG["staging"]
    STG -- "PR: ci-pass, check-source,<br/>1 approval" --> MAIN["main"]
    STG -. "on merge" .-> CDS["cd-staging<br/>build, smoke test,<br/>push to GHCR"]
    MAIN -. "on merge" .-> CDP["cd-prod<br/>deploy the verified image,<br/>live smoke test"]
    CDS -. "same tree hash" .-> CDP
    MAN(["manual<br/>workflow_dispatch"]) -.-> MW["ct-retrain / rollback<br/>move @champion, refresh"]
```

* **`ci-pass`** (`ci-pr.yml`, every PR into `dev`, `staging` or `main`): lockfile, actionlint, gitleaks, ruff, pytest, Docker build and container smoke test, PR title. A failing PR cannot merge.
* **`check-source`** (`enforce-source-branch.yml`): only `dev` → `staging` and `staging` → `main`.
* **`cd-staging`** (merge to `staging`): validates the release candidate against the real registry, then publishes it as `:<commit-sha>` and `:tree-<tree-hash>`.
* **`cd-prod`** (merge to `main`): deploys that exact image to Render by digest, waits until it is live, then smoke-tests the live URL with a real prediction and the expected `@champion` version.
* **`ct-retrain`** / **`rollback`** (manual): change the served model without a redeploy.

Full trigger table and reproduction steps: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#workflows-and-the-branches-that-trigger-them). Branch rules: [`CONTRIBUTING.md`](CONTRIBUTING.md#2-git-branching--pull-request-workflow).

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
