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

Put the variables listed in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#run-locally-with-docker) in a git-ignored `.env`. `MLFLOW_TRACKING_PASSWORD` is your DagsHub token, also used as the DVC remote password.

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
<type>/<topic> ──PR──► dev ──PR──► staging ──PR──► main
```

1. **Branch off `dev`** with a short-lived branch named per [Conventional Branch](https://conventional-branch.github.io/) (`feature/*`, `feat/*`, `fix/*`, `docs/*`, `ci/*`, ...).
2. **Write commits and PR titles** per [Conventional Commits](https://www.conventionalcommits.org/) (the PR title is checked in CI).
3. **Open PRs against `dev`**. Keep them small and focused.
4. **Promote releases** with a PR from `dev` to `staging`, then from `staging` to `main`. Any other source branch is rejected. Merging to `staging` validates the release candidate; merging to `main` deploys it (see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#workflows-and-the-branches-that-trigger-them)).

Repository rulesets protect `dev`, `staging` and `main`: changes go through a pull request with one approval, force-push and deletion are blocked, and the `ci-pass` check must pass (plus `check-source` on `staging` and `main`).

---

## 3. Code Quality, Testing & Docker Verification

Before opening a PR, run the checks enforced by `.github/workflows/ci-pr.yml`:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests use fake models and a mocked MLflow registry, so they never touch production models. Add tests next to the code you change: `tests/test_app.py` for the API (predictions, invalid inputs, model loading and refresh), `tests/test_policy*.py` and `tests/test_registry.py` for promotion and rollback rules.

To check the image, build and smoke-test it locally as described in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#run-locally-with-docker).

---

## 4. Data & Model Hygiene Rules

* **No datasets or model binaries in Git**: track data with DVC (`dvc add` → `dvc push`) and commit only the `.dvc` pointer files.
* **Preserve chronological splits**: data preparation must keep `max(train.date) < min(eval.date)` and fit preprocessing on the training split only.
* **Never edit the production `@champion` alias by hand**: promotions and rollbacks go through `ct-retrain.yml` or `rollback.yml` so they are logged in MLflow.
