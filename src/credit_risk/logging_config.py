"""Structured logging configured for Cloud Logging.

Cloud Run captures stdout. If a process writes plain text, every line lands in
Cloud Logging as an unstructured blob with severity ``DEFAULT`` - you cannot
filter by severity, and request correlation is lost. Emitting one JSON object per
line with the field names Cloud Logging recognises (``severity``, ``message``,
``logging.googleapis.com/trace``) makes logs queryable:

    severity>=ERROR AND jsonPayload.endpoint="/predict"

The same JSON is perfectly readable locally, so there is no separate dev format
to drift out of sync.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

# LogRecord attributes that are structural rather than user-supplied. Anything
# outside this set was passed via `extra=` and belongs in the JSON payload.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

# Python level names -> Cloud Logging severities. Only WARNING differs.
_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class CloudLoggingFormatter(logging.Formatter):
    """Render records as single-line JSON in Cloud Logging's expected shape."""

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self.project_id = project_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": _SEVERITY.get(record.levelname, record.levelname),
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Promote `extra=` fields to top level so they are queryable as
        # jsonPayload.<field> rather than buried in the message string.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        # Cloud Run injects the trace id here; surfacing it lets Cloud Logging
        # group every log line emitted while serving one request.
        trace_header = getattr(record, "trace_header", None)
        if trace_header and self.project_id:
            trace_id = trace_header.split("/")[0]
            payload["logging.googleapis.com/trace"] = (
                f"projects/{self.project_id}/traces/{trace_id}"
            )

        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    project_id: str | None = None,
) -> None:
    """Install the root logging handler. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Replace existing handlers so uvicorn's default text handler does not
    # double-print every line alongside our JSON.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(CloudLoggingFormatter(project_id=project_id))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
        )
    root.addHandler(handler)

    # These emit one INFO line per request/operation and drown the signal.
    for noisy in ("urllib3", "google.auth", "google.cloud", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_from_settings(settings: Any) -> None:
    """Configure logging from a :class:`~credit_risk.config.Settings` instance."""
    configure_logging(
        level=settings.logging.level,
        # Honour an explicit override so `CR__LOGGING__JSON=false` gives readable
        # console output during local debugging.
        json_format=settings.logging.json_output and os.environ.get("CR_PLAIN_LOGS") != "1",
        project_id=settings.gcp.project_id,
    )
