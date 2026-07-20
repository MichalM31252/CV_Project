"""Persist served predictions for monitoring.

Every prediction is written with its inputs, its score, the model version that
produced it and a timestamp. Without this table three routine questions have no
answer: has the input distribution moved since training, is the flag rate
drifting, and which model version produced a decision a customer is disputing.

Writes are buffered and flushed in batches. On BigQuery each insert is a billable
API call with meaningful latency, so writing per prediction would put warehouse
round-trips on the serving hot path. The buffer is deliberately best-effort:
monitoring is not allowed to fail a prediction, so flush errors are logged and
swallowed.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..config import Settings
from ..warehouse import get_warehouse

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SIZE = 50


class PredictionLogger:
    """Buffered writer for served predictions."""

    def __init__(self, settings: Settings, buffer_size: int = DEFAULT_BUFFER_SIZE) -> None:
        self.settings = settings
        self.buffer_size = buffer_size
        self.table = settings.table_ref(settings.data.predictions_table)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def log(self, records: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
        """Buffer predictions, flushing when the buffer fills."""
        now = datetime.now(UTC).replace(tzinfo=None)
        rows = [
            {
                "predicted_at": now,
                "client_id": result["client_id"],
                "model_version": result["model_version"],
                "default_probability": result["default_probability"],
                "decision": result["decision"],
                "risk_band": result["risk_band"],
                "threshold": result["threshold"],
                # A subset of inputs, kept so drift can be computed from this
                # table alone without joining back to a source system.
                "limit_bal": record.get("limit_bal"),
                "age": record.get("age"),
                "pay_status_1": record.get("pay_status_1"),
                "bill_amt_1": record.get("bill_amt_1"),
                "pay_amt_1": record.get("pay_amt_1"),
            }
            for record, result in zip(records, results, strict=True)
        ]

        with self._lock:
            self._buffer.extend(rows)
            should_flush = len(self._buffer) >= self.buffer_size
        if should_flush:
            self.flush()

    def flush(self) -> int:
        """Write buffered rows. Returns the number written; never raises."""
        with self._lock:
            if not self._buffer:
                return 0
            pending, self._buffer = self._buffer, []

        try:
            with get_warehouse(self.settings) as warehouse:
                warehouse.write_table(pd.DataFrame(pending), self.table, mode="append")
            logger.info("flushed predictions", extra={"rows": len(pending), "table": self.table})
            return len(pending)
        except Exception:  # noqa: BLE001 - monitoring must never break serving
            logger.exception("failed to flush prediction log", extra={"rows": len(pending)})
            return 0

    def close(self) -> None:
        self.flush()
