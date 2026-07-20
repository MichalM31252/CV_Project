"""BigQuery implementation of :class:`Warehouse` - the production backend.

Import of ``google.cloud.bigquery`` is deferred to construction time so that the
local backend never pays for the (fairly heavy) client import, and so that CI can
run the full test suite without the GCP extras installed.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import Warehouse, WriteMode

logger = logging.getLogger(__name__)


class BigQueryWarehouse(Warehouse):
    """Warehouse backed by BigQuery.

    Parameters
    ----------
    project_id:
        GCP project that owns the dataset and is billed for query bytes.
    dataset:
        BigQuery dataset name (created by Terraform, see ``terraform/``).
    location:
        Dataset region. Must match the dataset or BigQuery raises rather than
        silently reading from another region.
    """

    def __init__(self, project_id: str, dataset: str, location: str = "europe-west1") -> None:
        from google.cloud import bigquery  # noqa: PLC0415 - deliberate lazy import

        self._bigquery = bigquery
        self.project_id = project_id
        self.dataset = dataset
        self.location = location
        self.client = bigquery.Client(project=project_id, location=location)

    def _qualify(self, table: str) -> str:
        """Expand a bare table name to ``project.dataset.table``."""
        if table.count(".") >= 2:
            return table
        return f"{self.project_id}.{self.dataset}.{table.split('.')[-1]}"

    def execute(self, sql: str) -> None:
        job = self.client.query(sql, location=self.location)
        job.result()  # block; surfaces SQL errors here rather than downstream
        if job.total_bytes_processed is not None:
            logger.info(
                "bigquery statement complete",
                extra={"bytes_processed": job.total_bytes_processed, "job_id": job.job_id},
            )

    def query(self, sql: str) -> pd.DataFrame:
        job = self.client.query(sql, location=self.location)
        df = job.result().to_dataframe()
        if job.total_bytes_processed is not None:
            logger.info(
                "bigquery query complete",
                extra={
                    "bytes_processed": job.total_bytes_processed,
                    "rows": len(df),
                    "job_id": job.job_id,
                },
            )
        return df

    def write_table(self, df: pd.DataFrame, table: str, mode: WriteMode = "replace") -> None:
        disposition = (
            self._bigquery.WriteDisposition.WRITE_TRUNCATE
            if mode == "replace"
            else self._bigquery.WriteDisposition.WRITE_APPEND
        )
        job_config = self._bigquery.LoadJobConfig(
            write_disposition=disposition,
            # Let BigQuery infer types from the Arrow schema pandas produces;
            # explicit schemas live in the Terraform table definitions instead.
            autodetect=True,
        )
        job = self.client.load_table_from_dataframe(
            df, self._qualify(table), job_config=job_config, location=self.location
        )
        job.result()
        logger.info("loaded %d rows into %s", len(df), self._qualify(table))

    def table_exists(self, table: str) -> bool:
        from google.cloud.exceptions import NotFound  # noqa: PLC0415

        try:
            self.client.get_table(self._qualify(table))
        except NotFound:
            return False
        return True

    def close(self) -> None:
        self.client.close()
