"""Shared fixtures.

Tests run entirely on the DuckDB backend against synthetic data, so the suite
needs no GCP credentials, no network and no billing account - which is what lets
it run on every pull request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Settings, load_settings
from credit_risk.warehouse.duckdb_warehouse import DuckDBWarehouse


@pytest.fixture(scope="session")
def base_settings() -> Settings:
    return load_settings()


@pytest.fixture
def settings(tmp_path, base_settings) -> Settings:
    """Settings pointed at a per-test temporary directory.

    Copying the real config keeps tests honest - they exercise the same cost
    matrix, split bounds and feature list as production - while isolating all
    filesystem writes.
    """
    data = base_settings.model_dump()
    data["backend"] = "local"
    data["local"] = {
        "duckdb_path": tmp_path / "test.duckdb",
        "data_dir": tmp_path,
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "model_dir": tmp_path / "models",
    }
    # LoggingSettings uses an alias for `json`; dump/reload must agree on it.
    data["logging"] = {"level": "WARNING", "json": False}
    return Settings.model_validate(data)


@pytest.fixture
def warehouse(settings) -> DuckDBWarehouse:
    wh = DuckDBWarehouse(settings.local.duckdb_path)
    yield wh
    wh.close()


def make_raw_frame(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Synthetic data with the same schema and semantics as the source file.

    Generated rather than downloaded so tests stay fast, offline and
    deterministic. The target is deliberately correlated with delinquency and
    utilisation so that a model trained in tests has real signal to find - a
    pure-noise fixture would make any accuracy assertion meaningless.
    """
    rng = np.random.default_rng(seed)

    limit_bal = rng.choice([20_000, 50_000, 100_000, 200_000, 500_000], size=n)
    pay_status = rng.choice([-2, -1, 0, 1, 2, 3], size=(n, 6), p=[0.15, 0.2, 0.4, 0.13, 0.08, 0.04])
    bill = np.abs(rng.normal(0.4, 0.3, size=(n, 6)) * limit_bal[:, None]).round()
    pay = np.abs(rng.normal(0.1, 0.1, size=(n, 6)) * limit_bal[:, None]).round()

    # Risk rises with arrears and utilisation.
    utilisation = bill[:, 0] / limit_bal
    logit = -2.0 + 0.55 * pay_status[:, 0] + 1.2 * utilisation
    target = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    frame = pd.DataFrame(
        {
            "client_id": np.arange(1, n + 1),
            "limit_bal": limit_bal,
            "sex": rng.choice([1, 2], size=n),
            # Includes the undocumented codes (0, 5, 6) the clean stage collapses.
            "education": rng.choice([0, 1, 2, 3, 4, 5, 6], size=n),
            "marriage": rng.choice([0, 1, 2, 3], size=n),
            "age": rng.integers(21, 75, size=n),
            "default_next_month": target,
        }
    )
    for i in range(6):
        frame[f"pay_status_{i + 1}"] = pay_status[:, i]
        frame[f"bill_amt_{i + 1}"] = bill[:, i]
        frame[f"pay_amt_{i + 1}"] = pay[:, i]

    frame["ingested_at"] = pd.Timestamp("2026-01-01")
    frame["source_uri"] = "test://synthetic"
    return frame


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def feature_frame(settings, warehouse, raw_frame) -> pd.DataFrame:
    """Feature table built by the real SQL from synthetic raw data."""
    from credit_risk.features.build import build_features

    warehouse.write_table(raw_frame, settings.data.raw_table, mode="replace")
    return build_features(settings, warehouse)
