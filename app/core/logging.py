"""
Structured JSON Logging
-----------------------
Every log line is a JSON object. This makes it trivially ingestible by
Datadog, CloudWatch, Vercel log drains, or any log aggregator.

Fields emitted on every log:
  - timestamp   ISO-8601
  - level       DEBUG | INFO | WARNING | ERROR
  - logger      module path
  - message     human-readable description
  - **kwargs    all structured data passed by the caller
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra structured fields the caller passed via `extra=`
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_request_event(
    logger: logging.Logger,
    request_id: str,
    event: str,
    **kwargs: Any,
) -> None:
    """Helper that enforces a consistent request-scoped log shape."""
    logger.info(
        event,
        extra={"request_id": request_id, **kwargs},
    )
