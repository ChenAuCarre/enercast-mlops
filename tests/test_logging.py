import json
import logging
import sys

from app.logging import JsonFormatter, request_id_var


def test_json_line_carries_extras_request_id_and_exception():
    try:
        raise ZeroDivisionError("boom")
    except ZeroDivisionError:
        record = logging.makeLogRecord(
            {"name": "app", "msg": "loaded %s", "args": ("v2",), "model_version": "2"}
        )
        record.exc_info = sys.exc_info()
    token = request_id_var.set("rid-9")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert payload["event"] == "loaded v2"
    assert payload["model_version"] == "2"
    assert payload["request_id"] == "rid-9"
    assert "ZeroDivisionError" in payload["exc"]
