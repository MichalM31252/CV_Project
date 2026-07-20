"""Stage 2 - land the raw Parquet in the warehouse.

On GCP this is the standard two-hop ingestion pattern:

    local Parquet -> Cloud Storage -> BigQuery external load

Staging through GCS rather than streaming a DataFrame straight into BigQuery is
what a production job does, for three reasons: the object in GCS is an immutable,
auditable record of exactly what was loaded; BigQuery load jobs from GCS are free
whereas streaming inserts are billed; and a failed load can be replayed from the
staged object without re-fetching from the upstream source.

Locally the same function writes the frame into DuckDB, so the stage boundary and
the resulting table name are identical either way.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..storage import GCSBlobStore
from ..warehouse import Warehouse

logger = logging.getLogger(__name__)


def _stage_to_gcs(settings: Settings, parquet_path: Path) -> str:
    """Upload the Parquet file to the raw bucket under a dated prefix.

    Date-partitioned prefixes keep successive loads side by side, which is what
    makes point-in-time replay possible.
    """
    store = GCSBlobStore(settings.gcp.raw_bucket, prefix="raw/credit_default")
    remote = f"dt={datetime.now(UTC):%Y-%m-%d}/{parquet_path.name}"
    return store.upload(parquet_path, remote)


def load_raw_table(
    settings: Settings,
    warehouse: Warehouse,
    parquet_path: Path,
) -> int:
    """Load the raw Parquet into the warehouse raw table. Returns the row count."""
    table = settings.table_ref(settings.data.raw_table)

    if settings.backend == "gcp":
        uri = _stage_to_gcs(settings, parquet_path)
        logger.info("staged raw file to Cloud Storage", extra={"uri": uri})

    df = pd.read_parquet(parquet_path)

    # WRITE_TRUNCATE semantics: this stage is idempotent by design, so a rerun
    # after a failure produces the same table rather than doubling the rows.
    warehouse.write_table(df, table, mode="replace")

    count = warehouse.row_count(table)
    logger.info("raw table loaded", extra={"table": table, "rows": count})
    return count
