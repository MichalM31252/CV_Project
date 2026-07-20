"""Typed configuration for the credit-default pipeline.

Configuration resolution order (lowest to highest precedence):

1. ``config/config.yaml``
2. environment variables using ``CR__SECTION__KEY`` (double underscore = nesting)

Keeping one typed object shared by ingestion, training, serving and monitoring
means the API cannot silently disagree with the training job about which model
flavour or feature list is in play.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Repository root: src/credit_risk/config.py -> up three levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

Backend = Literal["local", "gcp"]


class GCPSettings(BaseModel):
    project_id: str | None = None
    region: str = "europe-west1"
    bq_dataset: str = "credit_risk"
    raw_bucket: str | None = None
    artifact_bucket: str | None = None
    artifact_repo: str = "credit-risk"
    service_name: str = "credit-risk-api"


class LocalSettings(BaseModel):
    duckdb_path: Path = Path("data/processed/warehouse.duckdb")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("data/models")

    @model_validator(mode="after")
    def _absolutise(self) -> LocalSettings:
        """Resolve relative paths against the repo root, not the CWD.

        Without this, running `python -m credit_risk.serving.app` from a
        subdirectory would silently create a second, empty data tree.
        """
        for field in type(self).model_fields:
            value = getattr(self, field)
            if isinstance(value, Path) and not value.is_absolute():
                object.__setattr__(self, field, PROJECT_ROOT / value)
        return self


class DataSettings(BaseModel):
    source_url: str
    source_member: str
    source_sha256: str
    raw_table: str = "raw_credit_default"
    clean_table: str = "clean_credit_default"
    features_table: str = "feature_credit_default"
    predictions_table: str = "prediction_log"
    drift_table: str = "drift_metrics"


class SplitSettings(BaseModel):
    hash_multiplier: int = 2654435761
    hash_modulus: int = 100
    train_max: int = 69
    valid_max: int = 84

    @model_validator(mode="after")
    def _check_ordering(self) -> SplitSettings:
        if not 0 <= self.train_max < self.valid_max < self.hash_modulus:
            raise ValueError(
                "split bounds must satisfy 0 <= train_max < valid_max < hash_modulus; "
                f"got train_max={self.train_max}, valid_max={self.valid_max}, "
                f"hash_modulus={self.hash_modulus}"
            )
        return self


class CostMatrix(BaseModel):
    """Business cost of each confusion-matrix cell, in New Taiwan dollars."""

    false_negative: float
    false_positive: float
    true_positive: float
    true_negative: float

    def expected_cost(self, tn: int, fp: int, fn: int, tp: int) -> float:
        return (
            tn * self.true_negative
            + fp * self.false_positive
            + fn * self.false_negative
            + tp * self.true_positive
        )


class SklearnSettings(BaseModel):
    n_iter_search: int = 25
    cv_folds: int = 4
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"


class TorchSettings(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10


class ModelSettings(BaseModel):
    target: str = "default_next_month"
    protected_attributes: list[str] = Field(default_factory=list)
    exclude_from_features: list[str] = Field(default_factory=list)
    random_state: int = 42
    cost: CostMatrix
    sklearn: SklearnSettings = Field(default_factory=SklearnSettings)
    torch: TorchSettings = Field(default_factory=TorchSettings)


class ServingSettings(BaseModel):
    model_name: str = "credit_default_classifier"
    model_flavor: Literal["sklearn", "torch"] = "sklearn"
    max_batch_size: int = 500


class MonitoringSettings(BaseModel):
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    tracked_features: list[str] = Field(default_factory=list)


class LoggingSettings(BaseModel):
    # Aliased because a field literally named `json` shadows BaseModel.json.
    # The YAML/env-var key stays `json`; the Python attribute is `json_output`.
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")


class Settings(BaseModel):
    backend: Backend = "local"
    project_name: str = "credit-default-risk"
    gcp: GCPSettings = Field(default_factory=GCPSettings)
    local: LocalSettings = Field(default_factory=LocalSettings)
    data: DataSettings
    split: SplitSettings = Field(default_factory=SplitSettings)
    model: ModelSettings
    serving: ServingSettings = Field(default_factory=ServingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def _require_gcp_fields(self) -> Settings:
        """Fail fast at startup rather than mid-pipeline on a half-set GCP config."""
        if self.backend == "gcp":
            missing = [
                name
                for name in ("project_id", "raw_bucket", "artifact_bucket")
                if getattr(self.gcp, name) in (None, "")
            ]
            if missing:
                raise ValueError(
                    "backend='gcp' requires gcp."
                    + ", gcp.".join(missing)
                    + ". Set them in config/config.yaml or via CR__GCP__<NAME> env vars."
                )
        return self

    # -- derived helpers ---------------------------------------------------

    def table_ref(self, table: str) -> str:
        """Fully-qualified table name for the active backend.

        BigQuery needs ``project.dataset.table``; DuckDB uses a bare name. Callers
        format SQL against this so the same statement targets either engine.
        """
        if self.backend == "gcp":
            return f"{self.gcp.project_id}.{self.gcp.bq_dataset}.{table}"
        return table

    @property
    def model_dir(self) -> Path:
        """Local directory holding model artifacts (also the staging dir for GCS)."""
        return self.local.model_dir


def _apply_env_overrides(raw: dict[str, Any], prefix: str = "CR__") -> dict[str, Any]:
    """Overlay ``CR__A__B=value`` environment variables onto the YAML mapping.

    Values are parsed as YAML so ``true``/``42``/``[a, b]`` arrive correctly typed
    instead of as strings.
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = [part.lower() for part in env_key[len(prefix) :].split("__") if part]
        if not path:
            continue
        cursor = raw
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        try:
            cursor[path[-1]] = yaml.safe_load(env_value)
        except yaml.YAMLError:
            cursor[path[-1]] = env_value
    return raw


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML plus environment overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(_apply_env_overrides(raw))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings.

    Cached because the FastAPI app resolves settings per request; re-reading and
    re-validating the YAML on every prediction would be wasted latency.
    """
    return load_settings(os.environ.get("CR_CONFIG_PATH"))
