"""Prometheus metrics exposed by the prediction API at ``/metrics``."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest

http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests received",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["path"],
)

predictions_total = Counter(
    "predictions_total",
    "Total number of predictions served by the champion",
    ["model_version"],
)

prediction_value_wh = Histogram(
    "prediction_value_wh",
    "Predicted appliance energy in Wh served by the champion",
    ["model_version"],
    buckets=(25, 50, 100, 200, 400, 800, 1600),  # defaults are for seconds
)

model_info = Gauge(
    "model_info",
    "Version of the loaded champion model (value is always 1)",
    ["model_version"],
)

feature_drift = Gauge(
    "feature_drift",
    "Drift of a feature against the champion training profile (PSI)",
    ["feature"],
)


def observe_http_request(
    method: str,
    path: str,
    status: int,
    duration_seconds: float,
) -> None:
    """Record one HTTP request (including the Prometheus scrapes of /metrics)."""
    http_requests_total.labels(method=method, path=path, status=str(status)).inc()
    http_request_duration_seconds.labels(path=path).observe(duration_seconds)


def observe_prediction(model_version: str, prediction_wh: float) -> None:
    """Record one served prediction."""
    predictions_total.labels(model_version=model_version).inc()
    prediction_value_wh.labels(model_version=model_version).observe(
        float(prediction_wh)
    )


def set_model_info(version: str) -> None:
    """Report the version of the loaded champion (no series until one loads)."""
    model_info.clear()  # drop the previous version's series on refresh
    model_info.labels(model_version=str(version)).set(1)


def set_feature_drift(feature: str, value: float) -> None:
    """Report the drift of one feature against the champion training profile."""
    feature_drift.labels(feature=feature).set(float(value))


def clear_feature_drift() -> None:
    """Drop all feature drift series (the reference they were computed on is gone)."""
    feature_drift.clear()


def render_metrics() -> bytes:
    """Return the Prometheus exposition payload."""
    return generate_latest()
