"""Inference service: raw account data in, calibrated risk decision out.

The critical design decision is here.

Features are engineered in ``sql/01_clean.sql`` and ``sql/02_features.sql`` and
computed in the warehouse at training time. Serving needs those same features
from raw request payloads. The usual approach is to reimplement the logic in
pandas for the API - and that is precisely how training/serving skew arises: two
implementations of one definition, drifting apart at the first change, producing
predictions that no longer match what the model was trained on, with no error to
alert anyone.

Instead, this class executes **the same SQL files** against an in-process DuckDB
instance holding just the request rows. The feature definitions have exactly one
implementation. A change to the SQL propagates to training and serving
simultaneously, because it is the same file.

The cost is a DuckDB dependency in the serving image and a few milliseconds per
request. That is a good trade for making an entire class of silent production
bugs structurally impossible.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, Settings
from ..training.registry import load_model

logger = logging.getLogger(__name__)

SQL_DIR = PROJECT_ROOT / "sql"

# Risk bands for downstream routing. Boundaries are relative to the operating
# threshold rather than fixed probabilities, so they stay meaningful if the cost
# matrix - and therefore the threshold - is revised.
LOW_BAND_FACTOR = 0.5
HIGH_BAND_FACTOR = 2.0


class Predictor:
    """Loads a registered model and scores raw client records."""

    def __init__(
        self, settings: Settings, flavor: str | None = None, version: str = "latest"
    ) -> None:
        self.settings = settings
        self.flavor = flavor or settings.serving.model_flavor
        self.model, self.metadata = load_model(settings, flavor=self.flavor, version=version)
        self.threshold: float = float(self.metadata["threshold"])
        self.feature_names: list[str] = list(self.metadata["feature_names"])
        self.version: str = str(self.metadata["version"])

        # One in-memory DuckDB connection reused across requests: creating one per
        # request would dominate latency. DuckDB connections are not thread-safe,
        # so a lock guards it - the SQL is sub-millisecond, so contention is not
        # the bottleneck under Cloud Run's per-instance concurrency.
        self._conn = duckdb.connect(":memory:")
        self._lock = threading.Lock()

        self._clean_sql = (SQL_DIR / "01_clean.sql").read_text(encoding="utf-8")
        self._features_sql = (SQL_DIR / "02_features.sql").read_text(encoding="utf-8")

        logger.info(
            "predictor ready",
            extra={"flavor": self.flavor, "version": self.version, "threshold": self.threshold},
        )

    # -- feature computation ------------------------------------------------

    def build_features(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Run the production feature SQL over request rows."""
        raw = pd.DataFrame(records)
        # The clean stage ranks by ingested_at for its de-duplication guard, so the
        # column has to be present even for a single scoring request.
        raw["ingested_at"] = datetime.now(UTC).replace(tzinfo=None)
        # Not supplied by the API and not used by any feature; present only to
        # satisfy the clean-stage projection.
        raw["default_next_month"] = 0

        params = {
            "raw_table": "request_raw",
            "clean_table": "request_clean",
            "features_table": "request_features",
            "hash_multiplier": self.settings.split.hash_multiplier,
            "hash_modulus": self.settings.split.hash_modulus,
            "train_max": self.settings.split.train_max,
            "valid_max": self.settings.split.valid_max,
        }

        with self._lock:
            self._conn.register("request_raw", raw)
            try:
                self._conn.execute(self._clean_sql.format(**params))
                self._conn.execute(self._features_sql.format(**params))
                features = self._conn.execute(
                    "SELECT * FROM request_features ORDER BY client_id"
                ).df()
            finally:
                self._conn.unregister("request_raw")

        if len(features) != len(raw):
            # The clean stage filters implausible rows (limit_bal <= 0, age out of
            # range). Schema validation should have caught these already, so
            # reaching here means the two disagree and that must not pass silently.
            raise ValueError(
                f"feature stage returned {len(features)} rows for {len(raw)} records; "
                "input violates a cleaning rule not covered by request validation"
            )
        return features

    # -- scoring ------------------------------------------------------------

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score records. Returns one result dict per record, ordered by client_id."""
        start = time.perf_counter()
        features = self.build_features(records)

        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"feature stage did not produce required columns: {missing}")

        # Reindex to the exact training column order. Passing columns in a
        # different order to a fitted ColumnTransformer silently scores garbage.
        model_input = features[self.feature_names]
        probabilities = self.model.predict_proba(model_input)[:, 1]

        results = [
            {
                "client_id": int(client_id),
                "default_probability": float(probability),
                "decision": "flag" if probability >= self.threshold else "no_flag",
                "threshold": self.threshold,
                "risk_band": self._risk_band(probability),
                "model_version": self.version,
            }
            for client_id, probability in zip(features["client_id"], probabilities, strict=True)
        ]

        logger.info(
            "prediction served",
            extra={
                "n_records": len(results),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "model_version": self.version,
                "flagged": sum(r["decision"] == "flag" for r in results),
            },
        )
        return results

    def _risk_band(self, probability: float) -> str:
        if probability < self.threshold * LOW_BAND_FACTOR:
            return "low"
        if probability < self.threshold * HIGH_BAND_FACTOR:
            return "medium"
        return "high"

    def feature_vector(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Engineered features for the given records - used by drift monitoring."""
        return self.build_features(records)

    def close(self) -> None:
        self._conn.close()


def probability_summary(probabilities: np.ndarray) -> dict[str, float]:
    """Compact distribution summary for logging and drift checks."""
    return {
        "mean": float(np.mean(probabilities)),
        "p50": float(np.percentile(probabilities, 50)),
        "p95": float(np.percentile(probabilities, 95)),
        "max": float(np.max(probabilities)),
    }
