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
