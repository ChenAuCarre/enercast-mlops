"""Tests for the structured JSON logging of the prediction API."""

import json
import logging

from app import config
from app import logging as app_logging


def _record(level=logging.INFO, msg="prediction_made", exc_info=None, **extra):
    record = logging.LogRecord("app", level, __file__, 1, msg, (), exc_info)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_one_json_object_per_line():
    line = app_logging.JsonFormatter().format(
        _record(
            msg="prediction_made", model_version="2", prediction=1.5, feature_count=27
        )
    )

    payload = json.loads(line)

    assert payload["event"] == "prediction_made"
    assert payload["level"] == "info"
    assert payload["logger"] == "app"
    assert payload["model_version"] == "2"
    assert payload["prediction"] == 1.5
    assert payload["feature_count"] == 27
    assert "timestamp" in payload


def test_json_formatter_includes_exceptions():
    formatter = app_logging.JsonFormatter()

    class Error(Exception):
        pass

    error = Error("boom")
    formatted = formatter.format(
        _record(
            level=logging.ERROR,
            msg="prediction_made_failed",
            exc_info=(Error, error, None),
        )
    )
    payload = json.loads(formatted)
    assert payload["level"] == "error"
    assert payload["event"] == "prediction_made_failed"
    assert "boom" in payload.get("exc", "")


def test_configure_logging_sets_the_app_logger_level():
    app_logging.configure_logging()

    expected_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    assert logging.getLogger("app").level == expected_level


def test_json_formatter_formats_message_args():
    record = logging.LogRecord(
        "app", logging.INFO, __file__, 1, "loaded %s", ("v2",), None
    )

    assert (
        json.loads(app_logging.JsonFormatter().format(record))["event"] == "loaded v2"
    )


def test_json_formatter_attaches_the_context_request_id():
    token = app_logging.set_request_id("rid-9")
    try:
        with_context = json.loads(app_logging.JsonFormatter().format(_record()))
        explicit = json.loads(
            app_logging.JsonFormatter().format(_record(request_id="other"))
        )
    finally:
        app_logging.reset_request_id(token)

    assert with_context["request_id"] == "rid-9"
    assert explicit["request_id"] == "other"


def test_configure_logging_routes_uvicorn_and_mlflow_logs_through_json():
    logging.getLogger("uvicorn").addHandler(logging.StreamHandler())
    logging.getLogger("uvicorn").propagate = False

    app_logging.configure_logging()
    app_logging.configure_logging()  # idempotent

    root_json = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h.formatter, app_logging.JsonFormatter)
    ]
    assert len(root_json) == 1
    for name in app_logging.JSON_ROUTED_LOGGERS:
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate
