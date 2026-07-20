"""Warehouse abstraction shared by the BigQuery and DuckDB backends.

The pipeline is written against this interface so that a single set of SQL
transformations can execute either against BigQuery in production or DuckDB on a
laptop / in CI. That is what keeps this repository runnable by anyone who clones
it without handing them a cloud bill, while the production path stays real
BigQuery code rather than a mock.

The SQL itself is deliberately restricted to constructs both engines share
(no ``SAFE_DIVIDE``, no backtick quoting, no ``FARM_FINGERPRINT``); division
guards use the portable ``x / NULLIF(y, 0)`` form instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd

WriteMode = Literal["replace", "append"]


class Warehouse(ABC):
    """Minimal warehouse surface: run SQL, read frames, write frames."""

    @abstractmethod
    def execute(self, sql: str) -> None:
        """Run a statement that returns no rows (DDL, CREATE TABLE AS, INSERT)."""

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        """Run a SELECT and return the result as a DataFrame."""

    @abstractmethod
    def write_table(self, df: pd.DataFrame, table: str, mode: WriteMode = "replace") -> None:
        """Persist a DataFrame as a table."""

    @abstractmethod
    def table_exists(self, table: str) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    # -- shared conveniences ------------------------------------------------

    def row_count(self, table: str) -> int:
        return int(self.query(f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0])

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
