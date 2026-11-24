FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Runtime deps only (no dev/experiment groups); cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group experiment --no-install-project

COPY app ./app
COPY ml ./ml

RUN useradd --system --no-create-home appuser
USER appuser

# The model is fetched from the MLflow registry at startup; configure via env:
# MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD, REFRESH_API_TOKEN.
# Cloud Run / Render inject PORT.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
