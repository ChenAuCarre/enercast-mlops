# Deployment

## Run locally with Docker

```bash
# create a git-ignored .env with the variables below
docker build -t enercast-api:local .
docker run --rm -p 8000:8000 --env-file .env enercast-api:local
curl localhost:8000/health  # {"status":"ready","model_loaded":true,"model_version":"..."}
```

Swagger UI: <http://localhost:8000/docs>. A sample `/predict` request is in the [README](../README.md).

Configuration is read from environment variables at start-up (`.env` is passed with `--env-file`); nothing is baked into the image:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking server and model registry (DagsHub) |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Registry credentials |
| `REFRESH_API_TOKEN` | Token for `POST /refresh` (the endpoint is disabled when unset) |
| `PORT` | Listen port, default `8000` (hosting platforms inject it) |
