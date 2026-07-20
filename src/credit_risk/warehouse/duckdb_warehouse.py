"""DuckDB implementation of :class:`Warehouse` - the local/CI backend.

DuckDB was chosen over SQLite or a pandas-only path because it speaks columnar
analytical SQL that is close enough to BigQuery standard SQL that the *same*
feature-engineering statements run unmodified on both. That is the property that
makes local development a genuine rehearsal of the production job.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .base import Warehouse, WriteMode


class DuckDBWarehouse(Warehouse):
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.database_path))

    def execute(self, sql: str) -> None:
        self._conn.execute(sql)

    def query(self, sql: str) -> pd.DataFrame:
        return self._conn.execute(sql).df()

    def write_table(self, df: pd.DataFrame, table: str, mode: WriteMode = "replace") -> None:
        # Registering the frame lets DuckDB read it zero-copy via Arrow rather
        # than round-tripping through INSERT statements.
        self._conn.register("_incoming", df)
        try:
            if mode == "replace":
                self._conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _incoming")
            else:
                if self.table_exists(table):
                    self._conn.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
                else:
                    self._conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming")
        finally:
            self._conn.unregister("_incoming")

    def table_exists(self, table: str) -> bool:
        # Strip any qualification so a BigQuery-style name still resolves locally.
        bare = table.split(".")[-1]
        result = self._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [bare],
        ).fetchone()
        return bool(result and result[0] > 0)

    def close(self) -> None:
        self._conn.close()
