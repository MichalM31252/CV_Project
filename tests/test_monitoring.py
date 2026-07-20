"""Tests for drift detection, prediction logging and configuration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Settings, load_settings
from credit_risk.monitoring.drift import (
    classify,
    compute_drift,
    population_stability_index,
)

# --------------------------- PSI --------------------------------------------


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 10000)
    psi, _ = population_stability_index(baseline, rng.normal(0, 1, 5000))
    assert psi < 0.01


def test_psi_increases_monotonically_with_shift():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 20000)
    values = [
        population_stability_index(baseline, rng.normal(shift, 1, 5000))[0]
        for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)


def test_large_shift_triggers_alert(settings):
    rng = np.random.default_rng(2)
    baseline = rng.normal(0, 1, 10000)
    psi, _ = population_stability_index(baseline, rng.normal(2.0, 1, 5000))
    assert classify(psi, settings) == "alert"


def test_disjoint_support_stays_finite():
    """Zero-mass bins would give ln(0) = -inf; epsilon flooring must prevent it."""
    rng = np.random.default_rng(3)
    psi, _ = population_stability_index(rng.normal(0, 1, 5000), rng.normal(100, 1, 1000))
    assert np.isfinite(psi)
    assert psi > 1.0


def test_empty_or_all_nan_input_returns_nan():
    rng = np.random.default_rng(4)
    baseline = rng.normal(0, 1, 1000)
    assert np.isnan(population_stability_index(baseline, np.array([]))[0])
    assert np.isnan(population_stability_index(baseline, np.full(100, np.nan))[0])


def test_constant_baseline_does_not_raise():
    rng = np.random.default_rng(5)
    psi, _ = population_stability_index(np.ones(500), rng.normal(0, 1, 500))
    assert np.isfinite(psi)


def test_nan_values_are_ignored_not_counted():
    rng = np.random.default_rng(6)
    baseline = rng.normal(0, 1, 5000)
    current = rng.normal(0, 1, 2000)
    with_nans = np.concatenate([current, np.full(500, np.nan)])
    clean_psi, _ = population_stability_index(baseline, current)
    nan_psi, _ = population_stability_index(baseline, with_nans)
    assert nan_psi == pytest.approx(clean_psi)


def test_breakdown_contributions_sum_to_psi():
    """The per-bin breakdown is what makes an alert actionable, so it must
    reconcile with the headline number."""
    rng = np.random.default_rng(7)
    psi, breakdown = population_stability_index(rng.normal(0, 1, 5000), rng.normal(0.5, 1, 5000))
    assert breakdown["psi_contribution"].sum() == pytest.approx(psi)


def test_compute_drift_flags_only_shifted_features(settings):
    rng = np.random.default_rng(8)
    n = 4000
    baseline = pd.DataFrame(
        {"age": rng.normal(40, 10, n), "limit_bal": rng.normal(150000, 50000, n)}
    )
    current = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),  # unchanged
            "limit_bal": rng.normal(400000, 50000, n),  # heavily shifted
        }
    )
    result = compute_drift(baseline, current, settings, features=["age", "limit_bal"])
    statuses = dict(zip(result["feature"], result["status"], strict=True))
    assert statuses["age"] == "stable"
    assert statuses["limit_bal"] == "alert"


def test_compute_drift_skips_missing_features(settings):
    baseline = pd.DataFrame({"age": [1.0, 2.0, 3.0]})
    result = compute_drift(
        baseline, pd.DataFrame({"age": [1.0, 2.0, 3.0]}), settings, features=["age", "absent"]
    )
    assert set(result["feature"]) == {"age"}


# --------------------------- prediction log ---------------------------------


def test_prediction_logger_flushes_to_warehouse(settings):
    from credit_risk.monitoring.prediction_log import PredictionLogger

    logger = PredictionLogger(settings, buffer_size=2)
    records = [
        {
            "client_id": i,
            "limit_bal": 1000.0,
            "age": 30,
            "pay_status_1": 0,
            "bill_amt_1": 100.0,
            "pay_amt_1": 50.0,
        }
        for i in (1, 2)
    ]
    results = [
        {
            "client_id": i,
            "model_version": "v1",
            "default_probability": 0.5,
            "decision": "flag",
            "risk_band": "medium",
            "threshold": 0.2,
        }
        for i in (1, 2)
    ]

    logger.log(records, results)  # reaching buffer_size triggers the flush

    from credit_risk.warehouse import get_warehouse

    with get_warehouse(settings) as warehouse:
        assert warehouse.row_count(settings.table_ref(settings.data.predictions_table)) == 2


def test_prediction_logger_never_raises_on_write_failure(settings, monkeypatch):
    """Monitoring must not be able to fail a prediction."""
    from credit_risk.monitoring import prediction_log as module

    def explode(*args, **kwargs):
        raise RuntimeError("warehouse unavailable")

    monkeypatch.setattr(module, "get_warehouse", explode)

    logger = module.PredictionLogger(settings, buffer_size=1)
    logger.log(
        [
            {
                "client_id": 1,
                "limit_bal": 1.0,
                "age": 1,
                "pay_status_1": 0,
                "bill_amt_1": 1.0,
                "pay_amt_1": 1.0,
            }
        ],
        [
            {
                "client_id": 1,
                "model_version": "v1",
                "default_probability": 0.5,
                "decision": "flag",
                "risk_band": "low",
                "threshold": 0.2,
            }
        ],
    )  # must not raise


# --------------------------- configuration ----------------------------------


def test_env_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("CR__MODEL__RANDOM_STATE", "123")
    monkeypatch.setenv("CR__SERVING__MAX_BATCH_SIZE", "7")
    settings = load_settings()
    assert settings.model.random_state == 123
    assert settings.serving.max_batch_size == 7


def test_gcp_backend_requires_project_configuration(monkeypatch):
    monkeypatch.setenv("CR__BACKEND", "gcp")
    with pytest.raises(ValueError, match="project_id"):
        load_settings()


def test_gcp_backend_qualifies_table_names(monkeypatch):
    monkeypatch.setenv("CR__BACKEND", "gcp")
    monkeypatch.setenv("CR__GCP__PROJECT_ID", "proj")
    monkeypatch.setenv("CR__GCP__RAW_BUCKET", "raw")
    monkeypatch.setenv("CR__GCP__ARTIFACT_BUCKET", "art")
    settings = load_settings()
    assert settings.table_ref("features") == "proj.credit_risk.features"


def test_local_backend_uses_bare_table_names(base_settings: Settings):
    assert base_settings.table_ref("features") == "features"


def test_split_bounds_must_be_ordered(base_settings: Settings):
    data = base_settings.model_dump()
    data["split"]["train_max"] = 90  # now above valid_max
    data["logging"] = {"level": "INFO", "json": True}
    with pytest.raises(ValueError, match="split bounds"):
        Settings.model_validate(data)


def test_cost_matrix_expected_cost():
    from credit_risk.config import CostMatrix

    cost = CostMatrix(false_negative=100, false_positive=10, true_positive=5, true_negative=0)
    assert cost.expected_cost(tn=1, fp=2, fn=3, tp=4) == 0 + 20 + 300 + 20
