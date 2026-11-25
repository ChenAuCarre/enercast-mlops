"""Structured JSON logging for the prediction API.

Every log line emitted by the app is one JSON object written to standard
output, so logs can be collected, searched and alerted on by tools such as
``docker compose logs`` + jq or Loki/ELK.

The log level comes from the ``LOG_LEVEL`` environment variable (default
``INFO``). The request id comes from the ``X-Request-ID`` header (or is
generated when missing) and is attached to every log line of the request.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app import config

# Attributes owned by logging.LogRecord; anything else found in the record is
# treated as structured context and copied into the JSON payload.
_RESERVED_LOG_RECORD_KEYS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "color_message",  # uvicorn's ANSI-coloured duplicate of the message
}

# Bound by the request middleware; attached to every log line of the request.
request_id_var: ContextVar[str] = ContextVar("app_request_id", default="")


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            "logger": record.name,
        }

        for key, value in vars(record).items():
            if key in _RESERVED_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            if callable(value):
                continue
            payload[key] = value

        # Context-bound id, unless the call site passed request_id explicitly.
        if "request_id" not in payload and (request_id := request_id_var.get()):
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# Libraries that install their own plain-text handlers.
# uvicorn.access is left alone so `--no-access-log` (Dockerfile) still silences it.
JSON_ROUTED_LOGGERS = ("app", "uvicorn", "uvicorn.error", "mlflow")


def configure_logging() -> None:
    """Send app, uvicorn and mlflow logs to stdout as JSON lines."""
    root = logging.getLogger()
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    for name in JSON_ROUTED_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("app").setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
