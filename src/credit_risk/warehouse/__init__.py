"""Warehouse factory - resolves the configured backend to a concrete client."""

from __future__ import annotations

from ..config import Settings
from .base import Warehouse, WriteMode
from .duckdb_warehouse import DuckDBWarehouse

__all__ = ["Warehouse", "WriteMode", "DuckDBWarehouse", "get_warehouse"]


def get_warehouse(settings: Settings) -> Warehouse:
    """Build the warehouse client for the configured backend.

    The BigQuery import stays inside the branch so the local path has no hard
    dependency on the Google Cloud SDK being importable.
    """
    if settings.backend == "gcp":
        from .bigquery_warehouse import BigQueryWarehouse  # noqa: PLC0415

        return BigQueryWarehouse(
            project_id=settings.gcp.project_id,  # validated non-None by Settings
            dataset=settings.gcp.bq_dataset,
            location=settings.gcp.region,
        )
    return DuckDBWarehouse(settings.local.duckdb_path)
