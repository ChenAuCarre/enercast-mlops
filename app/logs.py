import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        line = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id.get(),
        }
        if record.exc_info:
            line["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(line)


def setup(level):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # Route uvicorn's and mlflow's own handlers through the JSON root handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "mlflow"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)
