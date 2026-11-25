# Contributing to `enercast-mlops`

Thank you for contributing to `enercast-mlops`. This guide covers local environment setup, testing, Docker verification, the Git branching workflow, and ML hygiene standards.

---

## 1. Local Development Setup

### Prerequisites

* [uv](https://docs.astral.sh/uv/) (manages the pinned Python 3.12 runtime and dependencies)
* Docker (required for local container builds and smoke testing)
* Access to the project's DagsHub repository (required to pull DVC datasets or connect to the shared MLflow server)

### Install Dependencies

Clone the repository and synchronize the virtual environment:

```bash
git clone https://github.com/ChenAuCarre/enercast-mlops.git
cd enercast-mlops
uv sync
```

`pyproject.toml` configures `default-groups = ["dev", "experiment"]`, so `uv sync` installs runtime, development, and experimentation dependencies into `.venv`. To install a narrower set for a specific task:

```bash
uv sync --no-group experiment              # runtime + tests and linting (matches CI test job)
uv sync --no-dev --no-group experiment     # runtime only (matches the Docker image)
```

When adding a new dependency, place it in the appropriate group and commit both `pyproject.toml` and `uv.lock`:

* `uv add <pkg>` — runtime dependencies shipped in the production Docker image
* `uv add --dev <pkg>` — test and linting tools
* `uv add --group experiment <pkg>` — notebooks, DVC, and training dependencies

### Environment Variables & Secrets

Create a git-ignored `.env` file at the repository root when running the API container locally or reading from the shared MLflow server:

| Variable | Required | Description |
| :--- | :---: | :--- |
| `MLFLOW_TRACKING_URI` | Yes | DagsHub MLflow tracking and registry URL |
| `MLFLOW_TRACKING_USERNAME` | Yes | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | Yes | DagsHub personal access token (also used for the DVC remote) |
| `REFRESH_API_TOKEN` | No | Shared secret for `POST /refresh`; leaving it unset disables `/refresh` (`401`) |
| `PORT` | No | Listening port inside the container (default: `8000`) |

> **Security rule:** Never commit or paste credentials into Git commits, notebook outputs, logs, screenshots, issues, or pull requests. `.env` and `.dvc/config.local` are git-ignored. If a token is ever exposed, rotate it immediately.

### Pull Datasets with DVC

Datasets are tracked with DVC and stored on DagsHub. Store your DagsHub credentials locally in `.dvc/config.local` (never edit the committed `.dvc/config`), then pull the tracked snapshots:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <dagshub-username>
uv run dvc remote modify origin --local password <dagshub-token>
uv run dvc pull
```

---

## 2. Local Checks & Docker Verification

### Linting and Tests

Run the same checks enforced by `.github/workflows/ci-pr.yml` before opening a pull request:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Unit tests mock the MLflow registry and use fake models, so `uv run pytest` runs offline without touching the shared registry. Add or update tests alongside your changes:

* `tests/test_app.py` — API endpoints, validation errors, startup degradation, and atomic `/refresh` behavior
* `tests/test_preprocess.py` and `tests/test_splits.py` — feature engineering and chronological splitting
* `tests/test_train.py` and `tests/test_pipeline.py` — training runs, DVC hash checks, and pipeline orchestration
* `tests/test_policy.py`, `tests/test_policy_candidate.py`, and `tests/test_registry.py` — promotion gates, write guards, and rollback rules

### Build and Smoke-Test the Docker Image

Verify that the production container builds, loads `@champion`, and serves predictions locally:

```bash
docker build -t enercast-api:local .
docker run --rm -p 8000:8000 --env-file .env enercast-api:local
```

In a second terminal, verify readiness, metadata, and inference:

```bash
scripts/smoke_test.sh http://localhost:8000
```

Once running, the web UI is available at <http://localhost:8000/> and the interactive Swagger UI at <http://localhost:8000/docs>. See [`docs/API.md`](docs/API.md) for endpoint contracts and sample payloads.

---

## 3. Git Branching & Pull Request Workflow

Changes promote sequentially through three long-lived branches:

```text
<type>/<topic> ──PR──► dev ──PR──► staging ──PR──► main
```

1. **Branch from `dev`** using a short-lived branch named according to [Conventional Branch](https://conventional-branch.github.io/) (for example, `feat/*`, `feature/*`, `fix/*`, `docs/*`, `ci/*`).
2. **Follow [Conventional Commits](https://www.conventionalcommits.org/)** for commit messages and PR titles. The `pr-title` job in `ci-pr.yml` rejects non-conforming PR titles.
3. **Open feature PRs against `dev`**. Keep each PR small, focused, and easy to review.
4. **Promote releases in order**: open a PR from `dev` to `staging` to validate and publish the release-candidate image, then open a PR from `staging` to `main` to deploy that verified image to production. Any other source branch into `staging` or `main` is rejected by `enforce-source-branch.yml`.

### Branch Protection Rules

Repository rulesets protect `dev`, `staging`, and `main`. Force-pushes and branch deletions are disabled on all three branches, and every merge requires a pull request with:

* **1 approving review** (stale approvals are dismissed on new pushes)
* **All review threads resolved**
* **Passing `ci-pass` check**

Additional branch-specific rules:

| Target Branch | Allowed Merge Method | Extra Required Checks | Automated Action on Merge |
| :--- | :--- | :--- | :--- |
| `dev` | Squash merge only | — | None |
| `staging` | Merge commit only | `check-source` (`dev` → `staging`) | Runs `cd-staging.yml` (validates and pushes image to GHCR) |
| `main` | Merge commit only | `check-source` (`staging` → `main`) | Runs `cd-prod.yml` (deploys verified digest to Render) |

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full CI/CD pipeline specification.

---

## 4. Data & Model Hygiene Rules

* **Never commit raw datasets or model binaries to Git.** Track data files with DVC (`dvc add` → `dvc push`) and commit only the `.dvc` pointer files.
* **Preserve chronological splits.** Never shuffle time-series rows, always keep `max(train.date) < min(validation.date)`, keep `final_test.csv` isolated from model selection, and fit any stateful preprocessing exclusively on the training split. See [`ml/README_data.md`](ml/README_data.md) for the data contract.
* **Never edit the `@champion` alias manually.** All promotions and rollbacks must run through `ct-retrain.yml` or `rollback.yml` so MLflow records the policy decision and audit trail.
* **Never write to the shared registry from a local machine.** `register_model`, `promote_model`, and `rollback_champion` raise `LocalRegistryWriteForbidden` unless `GITHUB_ACTIONS=true`. For local experimentation, point MLflow to a local store (`export MLFLOW_TRACKING_URI="file:./mlruns"`) and run `uv run python -m ml.train`. See [`ml/README_experiments.md`](ml/README_experiments.md) for details.
