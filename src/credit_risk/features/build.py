"""Stage 3 - execute the SQL feature pipeline and validate the result.

The transformations themselves live in ``sql/`` as plain, reviewable SQL rather
than being buried in pandas. That choice is deliberate:

* the same statements run against BigQuery in production and DuckDB locally;
* feature definitions are reviewable by analysts who read SQL but not Python;
* computation happens next to the data, so the approach does not fall over when
  the table outgrows a single machine's memory.

This module supplies parameters, runs the statements in order, and then asserts
the properties the rest of the pipeline depends on. The validation is the point:
a feature table that is silently wrong is far more expensive than one that fails
loudly at build time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, Settings
from ..warehouse import Warehouse

logger = logging.getLogger(__name__)

SQL_DIR = PROJECT_ROOT / "sql"


class FeatureValidationError(RuntimeError):
    """Raised when the built feature table violates an expected invariant."""


def _render(path: Path, params: dict[str, object]) -> str:
    sql = path.read_text(encoding="utf-8")
    try:
        return sql.format(**params)
    except KeyError as exc:  # pragma: no cover - developer error, not runtime
        raise KeyError(f"{path.name} references unknown placeholder {exc}") from exc


def build_features(settings: Settings, warehouse: Warehouse, validate: bool = True) -> pd.DataFrame:
    """Run the clean and feature SQL stages. Returns the feature table.

    ``validate`` gates the whole-table invariant checks (all three splits
    present, non-degenerate target). They are correct for a pipeline run over the
    full dataset but meaningless for a handful of rows, so unit tests that
    exercise a single transformation switch them off.
    """
    params = {
        "raw_table": settings.table_ref(settings.data.raw_table),
        "clean_table": settings.table_ref(settings.data.clean_table),
        "features_table": settings.table_ref(settings.data.features_table),
        "hash_multiplier": settings.split.hash_multiplier,
        "hash_modulus": settings.split.hash_modulus,
        "train_max": settings.split.train_max,
        "valid_max": settings.split.valid_max,
    }

    for script in ("01_clean.sql", "02_features.sql"):
        logger.info("executing sql stage", extra={"script": script})
        warehouse.execute(_render(SQL_DIR / script, params))

    features = warehouse.query(f"SELECT * FROM {params['features_table']}")
    if validate:
        validate_features(features, settings)
    logger.info(
        "feature table built",
        extra={"rows": len(features), "columns": len(features.columns)},
    )
    return features


def validate_features(df: pd.DataFrame, settings: Settings) -> None:
    """Assert the invariants downstream stages rely on.

    Cheap to run and it converts a whole class of silent modelling bugs -
    a split that leaks, an all-null feature, a target that vanished - into an
    immediate, named failure.
    """
    problems: list[str] = []
    target = settings.model.target

    if df.empty:
        raise FeatureValidationError("feature table is empty")

    if target not in df.columns:
        problems.append(f"target column '{target}' missing")
    else:
        labels = set(df[target].dropna().unique())
        if not labels <= {0, 1}:
            problems.append(f"target must be binary, found values {sorted(labels)}")
        positive_rate = float(df[target].mean())
        if not 0.01 < positive_rate < 0.99:
            problems.append(f"degenerate target: positive rate {positive_rate:.4f}")

    if "client_id" in df.columns and df["client_id"].duplicated().any():
        problems.append(f"{int(df['client_id'].duplicated().sum())} duplicate client_id values")

    if "split_name" in df.columns:
        splits = set(df["split_name"].unique())
        if splits != {"train", "valid", "test"}:
            problems.append(f"expected splits train/valid/test, found {sorted(splits)}")

    # A column that is entirely null means an upstream join or a division guard
    # silently removed the signal.
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        problems.append(f"columns are entirely null: {all_null}")

    # Infinities break scikit-learn with an unhelpful error much later on.
    numeric = df.select_dtypes(include="number")
    infinite = [
        c
        for c in numeric.columns
        if not pd.Series(numeric[c])
        .replace([float("inf"), float("-inf")], pd.NA)
        .notna()
        .eq(numeric[c].notna())
        .all()
    ]
    if infinite:
        problems.append(f"columns contain infinite values: {infinite}")

    if problems:
        raise FeatureValidationError("feature validation failed:\n  - " + "\n  - ".join(problems))


def split_frames(
    df: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the feature table into train/valid/test frames."""
    return (
        df[df["split_name"] == "train"].reset_index(drop=True),
        df[df["split_name"] == "valid"].reset_index(drop=True),
        df[df["split_name"] == "test"].reset_index(drop=True),
    )


def feature_columns(df: pd.DataFrame, settings: Settings) -> list[str]:
    """Model input columns: everything except identifiers, target and excluded."""
    excluded = set(settings.model.exclude_from_features)
    return [c for c in df.columns if c not in excluded]
