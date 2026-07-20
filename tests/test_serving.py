"""Tests for the prediction API.

The important one is ``test_serving_features_match_warehouse_features``: it
pins the property the whole serving design exists to guarantee - that features
computed at request time are identical to those computed during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from credit_risk.serving.schemas import BatchPredictionRequest, ClientRecord

VALID_RECORD = {
    "client_id": 1,
    "limit_bal": 20000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 24,
    "pay_status_1": 2,
    "pay_status_2": 2,
    "pay_status_3": -1,
    "pay_status_4": -1,
    "pay_status_5": -2,
    "pay_status_6": -2,
    "bill_amt_1": 3913,
    "bill_amt_2": 3102,
    "bill_amt_3": 689,
    "bill_amt_4": 0,
    "bill_amt_5": 0,
    "bill_amt_6": 0,
    "pay_amt_1": 0,
    "pay_amt_2": 689,
    "pay_amt_3": 0,
    "pay_amt_4": 0,
    "pay_amt_5": 0,
    "pay_amt_6": 0,
}


# --------------------------- schema validation ------------------------------


def test_valid_record_accepted():
    assert ClientRecord(**VALID_RECORD).client_id == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 5),  # below the documented minimum
        ("age", 150),
        ("limit_bal", 0),  # zero limit makes every ratio undefined
        ("limit_bal", -5000),
        ("pay_status_1", 15),  # scale tops out at 9
        ("pay_status_1", -5),
        ("education", 99),
        ("pay_amt_1", -100),  # a payment cannot be negative
    ],
)
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        ClientRecord(**{**VALID_RECORD, field: value})


def test_negative_bill_allowed():
    """Overpayment leaves a credit balance - a real state, not an error."""
    assert ClientRecord(**{**VALID_RECORD, "bill_amt_1": -500}).bill_amt_1 == -500


def test_batch_rejects_duplicate_client_ids():
    with pytest.raises(ValidationError, match="unique"):
        BatchPredictionRequest(records=[ClientRecord(**VALID_RECORD), ClientRecord(**VALID_RECORD)])


def test_batch_rejects_empty():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(records=[])


# --------------------------- live app ---------------------------------------
#
# These need a trained model in the registry. The pipeline produces one, so they
# are skipped rather than failed when run on a clean checkout.


@pytest.fixture(scope="module")
def client():
    from credit_risk.config import get_settings
    from credit_risk.training.registry import load_model

    settings = get_settings()
    try:
        load_model(settings, flavor="sklearn")
    except FileNotFoundError:
        pytest.skip("no trained model available; run `make pipeline` first")

    from credit_risk.serving.app import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_ready_returns_200(client):
    assert client.get("/ready").status_code == 200


def test_model_info_exposes_lineage(client):
    body = client.get("/model-info").json()
    assert body["n_features"] == len(body["feature_names"])
    assert 0 < body["threshold"] < 1
    assert "test" in body["metrics"]
    # The protected attribute must not be among the model inputs.
    assert "sex" not in body["feature_names"]


def test_predict_returns_calibrated_probability(client):
    body = client.post("/predict", json=VALID_RECORD).json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"flag", "no_flag"}
    assert body["risk_band"] in {"low", "medium", "high"}
    # The decision must be consistent with the reported threshold.
    assert (body["default_probability"] >= body["threshold"]) == (body["decision"] == "flag")


def test_delinquent_client_scores_above_pristine_client(client):
    """Basic monotonicity: sustained arrears must not score lower than a client
    who pays in full every month."""

    def profile(client_id, status, bill, pay):
        record = dict(VALID_RECORD, client_id=client_id, limit_bal=200000)
        for i in range(1, 7):
            record[f"pay_status_{i}"] = status
            record[f"bill_amt_{i}"] = bill
            record[f"pay_amt_{i}"] = pay
        return record

    pristine = profile(1, -1, 5000, 5000)
    delinquent = profile(2, 3, 190000, 0)

    body = client.post("/predict/batch", json={"records": [pristine, delinquent]}).json()
    by_id = {p["client_id"]: p["default_probability"] for p in body["predictions"]}
    assert by_id[2] > by_id[1]


def test_batch_preserves_every_record(client):
    records = [dict(VALID_RECORD, client_id=i) for i in range(1, 11)]
    body = client.post("/predict/batch", json={"records": records}).json()
    assert len(body["predictions"]) == 10
    assert {p["client_id"] for p in body["predictions"]} == set(range(1, 11))


def test_batch_over_limit_rejected(client):
    from credit_risk.config import get_settings

    limit = get_settings().serving.max_batch_size
    records = [dict(VALID_RECORD, client_id=i) for i in range(limit + 5)]
    assert client.post("/predict/batch", json={"records": records}).status_code == 413


def test_malformed_request_returns_422(client):
    assert client.post("/predict", json={**VALID_RECORD, "age": 3}).status_code == 422


def test_serving_features_match_warehouse_features(client):
    """The core anti-skew guarantee.

    Features computed from a request must equal, to floating-point tolerance,
    those computed by the warehouse during training. If this ever fails, the API
    is scoring inputs the model was not trained on.
    """
    from credit_risk.config import get_settings
    from credit_risk.serving.predictor import Predictor
    from credit_risk.warehouse import get_warehouse

    settings = get_settings()
    with get_warehouse(settings) as warehouse:
        if not warehouse.table_exists(settings.data.clean_table):
            pytest.skip("warehouse not populated; run `make pipeline` first")
        clean = warehouse.query(
            f"SELECT * FROM {settings.data.clean_table} ORDER BY client_id LIMIT 200"  # noqa: S608
        )
        warehouse_features = warehouse.query(
            f"SELECT * FROM {settings.data.features_table} ORDER BY client_id LIMIT 200"  # noqa: S608
        )

    columns = [c for c in clean.columns if c not in ("ingested_at", "default_next_month")]
    predictor = Predictor(settings)
    try:
        served = predictor.build_features(clean[columns].to_dict("records"))
    finally:
        predictor.close()

    served = served.sort_values("client_id").reset_index(drop=True)
    expected = warehouse_features.sort_values("client_id").reset_index(drop=True)

    for column in predictor.feature_names:
        assert np.allclose(
            pd.to_numeric(served[column], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float),
            equal_nan=True,
            atol=1e-9,
        ), f"serving/training skew detected in feature '{column}'"
