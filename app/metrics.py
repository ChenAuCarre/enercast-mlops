"""Prometheus metrics exposed by the prediction API at ``/metrics``."""

from prometheus_client import Counter, Gauge, Histogram

http_requests_total = Counter(
    "http_requests_total", "HTTP requests", ["method", "path", "status"]
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request latency in seconds", ["path"]
)
predictions_total = Counter(
    "predictions_total", "Predictions served", ["model_version"]
)
prediction_value_wh = Histogram(
    "prediction_value_wh",
    "Predicted appliance energy in Wh",
    ["model_version"],
    buckets=(25, 50, 100, 200, 400, 800, 1600),  # defaults are for seconds
)
model_info = Gauge(
    "model_info", "Loaded champion version (value is always 1)", ["model_version"]
)
feature_drift = Gauge(
    "feature_drift",
    "PSI of a feature against the champion training profile",
    ["feature"],
)
