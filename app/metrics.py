from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)
PREDICTIONS = Counter("predictions_total", "Predictions served", ["model_version"])
SERVED_VERSION = Gauge("model_served_version", "Champion model version served")
DRIFT = Gauge("feature_drift_psi", "PSI vs. training distribution", ["feature"])
