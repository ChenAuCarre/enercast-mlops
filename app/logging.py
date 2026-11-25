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
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app import config

# Attributes owned by logging.LogRecord; anything else found in the record is
# treated as structured context and copied into the JSON payload.
_RESERVED_LOG_RECORD_KEYS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
}

_request_id_context: ContextVar[str] = ContextVar("app_request_id", default="")


def set_request_id(request_id: str) -> Any:
    """Bind a request id to the current context (middleware responsibility)."""
    return _request_id_context.set(request_id)


def reset_request_id(token: Any) -> None:
    """Undo a :func:`set_request_id` call."""
    _request_id_context.reset(token)


def get_request_id() -> str:
    """Return the request id bound to the current context ("" when unbound)."""
    return _request_id_context.get()


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
        if "request_id" not in payload and (request_id := get_request_id()):
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Attach the JSON formatter to the app logger and set the level."""
    logger = logging.getLogger("app")
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False


def get_logger(name: str = "app") -> logging.Logger:
    """Return a logger that emits structured JSON lines."""
    return logging.getLogger(name)
