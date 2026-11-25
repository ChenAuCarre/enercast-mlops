# Deployment

Production runs on **Render** as a Docker web service: <https://enercast-mlops-tree.onrender.com>.
The image contains the FastAPI app and the inference code only; the model is pulled from the
DagsHub MLflow registry (`wattwise-energy-forecaster@champion`) when the process starts or on
`POST /refresh`.

## Run locally with Docker

```bash
# create a git-ignored .env with the variables below
docker build -t enercast-api:local .
docker run --rm -p 8000:8000 --env-file .env enercast-api:local
curl localhost:8000/health  # {"status":"ready","model_loaded":true,"model_version":"..."}
scripts/smoke_test.sh http://localhost:8000  # readiness + /model-info + real prediction
```

Web UI: <http://localhost:8000/>. Swagger UI: <http://localhost:8000/docs>. A sample `/predict`
request is in [API.md](API.md#example-request).

Configuration is read from environment variables at start-up (`.env` is passed with `--env-file`); nothing is baked into the image:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking server and model registry (DagsHub) |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Registry credentials |
| `REFRESH_API_TOKEN` | Token for `POST /refresh` (the endpoint is disabled when unset) |
| `PORT` | Listen port, default `8000` (hosting platforms inject it) |
| `LOG_LEVEL` | Level of the JSON logs on stdout, default `INFO`. The container runs uvicorn with `--no-access-log`: the app logs each request once as `http_request`. |

## Workflows and the branches that trigger them

| Workflow / job | Trigger | Branch | What it does |
|---|---|---|---|
| `ci-pr.yml` (`ci-pass` + its jobs) | `pull_request` | into `dev`, `staging`, `main` | Lockfile check, actionlint, gitleaks, ruff, pytest, Docker build, container smoke test (`scripts/smoke_test.sh` against the built image), PR title. `ci-pass` is the required check. |
| `enforce-source-branch.yml` (`check-source`) | `pull_request` | into `staging`, `main` | Only `dev` → `staging` and `staging` → `main` are allowed. |
| `cd-staging.yml` (`validate`) | `push` (merge) | `staging` (also `workflow_dispatch`; publishes only on `staging`) | Builds the release candidate, runs it in Actions with the real registry, checks it serves the current `@champion` version with a real prediction, then pushes it to GHCR as `ghcr.io/chenaucarre/enercast-mlops:<commit-sha>` and `:tree-<tree-hash>`. |
| `cd-prod.yml` (`deploy`) | `push` (merge) | `main` (also `workflow_dispatch`) | Selects the image verified in staging by the tree hash of the merged commit (fails if none, so unvalidated code is never deployed), deploys it **by digest** through the Render deploy hook, polls the Render API until the deploy is `live`, then runs `scripts/smoke_test.sh` against the live URL with the expected `@champion` version. |
| `ct-retrain.yml` | `workflow_dispatch` (`dataset-version`, `force-initial-training`) | `dev`, `staging` or `main` | Continuous training: DVC pull + data contract validation, champion scoring, training, gated promotion, `/refresh` and live verification. `force-initial-training` allows a deliberate `v1` rebuild when a champion already exists. See [`ml/README_experiments.md`](../ml/README_experiments.md). |
| `rollback.yml` | `workflow_dispatch` (`target-version`, `reason`) | `dev`, `staging` or `main` | Moves `@champion` back to an approved version, `/refresh`, and verifies it with a live prediction. |

The tree tag exists because the `staging` → `main` merge creates a new commit SHA but keeps the
same tree, so `cd-prod` can deploy exactly the bytes `cd-staging` tested.

### GitHub settings used by the workflows

| Kind | Name | Used by |
|---|---|---|
| Variable | `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` | all |
| Variable | `API_BASE_URL` (live service URL) | `cd-prod`, `ct-retrain`, `rollback` |
| Secret | `MLFLOW_TRACKING_PASSWORD`, `REFRESH_API_TOKEN` | all (`MLFLOW_TRACKING_PASSWORD` also authenticates the DVC remote in `ct-retrain`) |
| Secret | `RENDER_DEPLOY_HOOK_URL`, `RENDER_API_KEY` | `cd-prod` |

The Render service itself holds `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
`MLFLOW_TRACKING_PASSWORD` and `REFRESH_API_TOKEN` as environment variables; Render injects `PORT`.
`REFRESH_API_TOKEN` must be the same value in Render and in GitHub.

## Reproduce a deployment

**Normal path:** merge `dev` → `staging` (wait for `cd-staging` to publish the image), then
`staging` → `main`. `cd-prod` deploys and smoke-tests automatically.

**Re-run:** `gh workflow run cd-prod.yml --ref main` redeploys the verified image for the current
`main` tree.

**By hand** (same steps as `cd-prod`):

```bash
image=ghcr.io/chenaucarre/enercast-mlops
digest=$(docker buildx imagetools inspect "$image:tree-$(git rev-parse 'main^{tree}')" \
  --format '{{json .Manifest}}' | jq -r .digest)
curl -X POST "$RENDER_DEPLOY_HOOK_URL&imgURL=$(jq -rn --arg v "$image@$digest" '$v|@uri')"
SMOKE_TIMEOUT=300 scripts/smoke_test.sh https://enercast-mlops-tree.onrender.com \
  "$(uv run python -c 'from ml.registry import get_champion_version; print(get_champion_version())')"
```

To roll back the **code**, redeploy an earlier verified image the same way (its `<commit-sha>`
tag or digest), or use Render's "Rollback" on a previous deploy. To roll back the **model**, use
`rollback.yml` (no redeploy needed).

## When the app does not become ready after a model change

- The service never reports ready without a model: `/health` is `200` only once a champion is
  loaded, otherwise `503`.
- A failed `/refresh` returns `502` and **keeps serving the previously loaded model**, so a broken
  new version does not take the service down. `ct-retrain` then fails its verification step and
  reports the previous champion version.
- A restart or redeploy while the alias points to a broken version starts the process with
  `/health` `503`; `cd-prod`'s smoke test then fails.
- Recovery: dispatch `rollback.yml` with the previous champion version and a reason. It restores
  the alias, refreshes the service and verifies it with a live prediction. If the service still
  fails on the restored version, the fault is the deployment, not the model: redeploy the last
  known-good image (see above).
