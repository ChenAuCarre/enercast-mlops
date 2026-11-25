"""JSON log lines on stdout for the ``app`` logger, tagged with the request id."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from app import config

# LogRecord's own attributes; anything else on a record came from `extra=`.
_RECORD_KEYS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Set by the request middleware for the duration of each request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            "logger": record.name,
        }
        payload |= {k: v for k, v in vars(record).items() if k not in _RECORD_KEYS}
        if request_id := request_id_var.get():
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging():
    logger = logging.getLogger("app")
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
