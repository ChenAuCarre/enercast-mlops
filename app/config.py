import os

# MLflow itself reads MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD from the env.
REFRESH_API_TOKEN = os.getenv("REFRESH_API_TOKEN", "")
