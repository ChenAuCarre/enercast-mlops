# Contributing to `enercast-mlops`

This repo is a production MLOps reference, and PR history tracks each contributor's work. All changes must follow the branching, testing, and data/model hygiene rules below.

---

## 1. Local Development Setup

### Prerequisites
* [uv](https://docs.astral.sh/uv/) (manages Python and dependencies)
* Docker (for local container smoke testing)
* Access to the team's DagsHub project (MLflow server and DVC remote)

### Installation

```bash
git clone https://github.com/ChenAuCarre/enercast-mlops.git
cd enercast-mlops
uv sync  # installs the pinned Python version and all dependencies into .venv
```

Install only what a task needs:

```bash
uv sync --no-group experiment                  # tests and linting, as in CI
uv sync --no-dev --no-group experiment         # runtime only, as in the Docker image
```

Add dependencies to the right group and commit the updated `pyproject.toml` and `uv.lock`:
* `uv add <pkg>`: runtime, shipped in the Docker image
* `uv add --dev <pkg>`: tests and linting
* `uv add --group experiment <pkg>`: notebooks and training

### Environment Variables & Secrets

Required variables:
* `MLFLOW_TRACKING_URI`
* `MLFLOW_TRACKING_USERNAME`
* `MLFLOW_TRACKING_PASSWORD` (the DagsHub token, also used as the DVC remote password)
* `REFRESH_API_TOKEN`

**Never expose secrets** in commits, logs, notebook outputs, screenshots, issues, or PR descriptions. If a secret leaks, rotate it immediately.

### DVC Remote

Datasets are versioned with DVC and stored on DagsHub. To download them, store DagsHub credentials in the gitignored `.dvc/config.local` (never in the committed `.dvc/config`), then pull:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <dagshub-username>
uv run dvc remote modify origin --local password <dagshub-token>
uv run dvc pull  # fetches the tracked datasets
```

---

## 2. Git Branching & Pull Request Workflow

Three long-lived branches, promoted in order:

```text
<type>/<topic> ──► dev ──► staging ──► main
```

1. **Branch off `dev`** and name branches per [Conventional Branch](https://conventional-branch.github.io/).
2. **Write commits** per [Conventional Commits](https://www.conventionalcommits.org/).
3. **Open PRs against `dev`**. Keep them small and focused. Merging requires one teammate's approval and passing `ci-pr.yml` checks.
4. **Promote releases**: `dev` → `staging` runs staging validation (`cd-staging.yml`); `staging` → `main` deploys to production (`cd-prod.yml`).

---

## 3. Code Quality, Testing & Docker Verification

Before opening a PR, run the checks enforced by `.github/workflows/ci-pr.yml`:

```bash
uv run ruff check app/ tests/
uv run pytest -v
```

Tests use fixtures and an isolated mock MLflow registry, so they never touch production models. New features need tests covering:
* Prediction schema and `@champion` version reporting (`tests/test_api_predictions.py`)
* Malformed or out-of-range inputs (`tests/test_input_validation.py`)
* Registry alias isolation and promotion gates (`tests/test_registry_and_gates.py`)

Verify the slim image builds without datasets, credentials, or dev-only packages:

```bash
docker build -t enercast-api:local .
docker run --rm -p 8000:8000 --env-file .env enercast-api:local
```

---

## 4. Data & Model Hygiene Rules

* **No datasets or model binaries in Git**: track data with DVC (`dvc add` → `dvc push`) and commit only the `.dvc` pointer files.
* **Preserve chronological splits**: data preparation must keep `max(train.date) < min(eval.date)` and fit preprocessing on the training split only.
* **Never edit the production `@champion` alias by hand**: promotions and rollbacks go through `ct-retrain.yml` or `rollback.yml` so they are logged in MLflow.
