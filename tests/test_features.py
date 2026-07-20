"""Tests for the SQL feature pipeline.

These assert the *semantics* of the transformations, not just that the SQL runs.
Each check corresponds to a decision documented in ``sql/02_features.sql`` - so
if someone later "simplifies" one of them, the reason it existed is recoverable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.build import (
    FeatureValidationError,
    feature_columns,
    split_frames,
    validate_features,
)


def test_feature_table_has_one_row_per_client(feature_frame, raw_frame):
    assert len(feature_frame) == len(raw_frame)
    assert not feature_frame["client_id"].duplicated().any()


def test_education_collapses_undocumented_codes(settings, warehouse, feature_frame):
    """Codes 0, 5 and 6 are absent from the codebook and map to 4 ('other')."""
    clean = warehouse.query(f"SELECT * FROM {settings.data.clean_table}")
    assert set(clean["education"].unique()) <= {1, 2, 3, 4}


def test_marriage_collapses_zero(settings, warehouse, feature_frame):
    clean = warehouse.query(f"SELECT * FROM {settings.data.clean_table}")
    assert set(clean["marriage"].unique()) <= {1, 2, 3}


def test_payment_ratio_uses_previous_month_bill(settings, warehouse):
    """pay_amt_i settles bill_amt_(i+1), not bill_amt_i.

    Guards the temporal alignment. Dividing by the same month's bill would be a
    lookahead: that statement had not been issued when the payment was made.
    """
    row = pd.DataFrame(
        {
            "client_id": [1],
            "limit_bal": [100_000],
            "sex": [1],
            "education": [2],
            "marriage": [1],
            "age": [40],
            "default_next_month": [0],
            "ingested_at": [pd.Timestamp("2026-01-01")],
            # bill_amt_2 = 1000 is what pay_amt_1 = 500 pays down -> ratio 0.5.
            # bill_amt_1 = 9999 must not appear in that ratio.
            "bill_amt_1": [9999],
            "bill_amt_2": [1000],
            "bill_amt_3": [2000],
            "bill_amt_4": [0],
            "bill_amt_5": [0],
            "bill_amt_6": [0],
            "pay_amt_1": [500],
            "pay_amt_2": [1000],
            "pay_amt_3": [0],
            "pay_amt_4": [0],
            "pay_amt_5": [0],
            "pay_amt_6": [0],
        }
    )
    for i in range(1, 7):
        row[f"pay_status_{i}"] = 0

    from credit_risk.features.build import build_features

    warehouse.write_table(row, settings.data.raw_table, mode="replace")
    features = build_features(settings, warehouse, validate=False)

    # avg over months 1..5 of pay_amt_i / bill_amt_(i+1), NULL treated as 0:
    #   500/1000=0.5, 1000/2000=0.5, 0/0->0, 0/0->0, 0/0->0  => 0.2
    assert features["avg_payment_ratio"].iloc[0] == pytest.approx(0.2)


def test_dormant_account_yields_null_payment_ratio_and_flag(settings, warehouse):
    """No statements in months 2-6 means the ratio is undefined, not zero."""
    row = pd.DataFrame(
        {
            "client_id": [1],
            "limit_bal": [100_000],
            "sex": [1],
            "education": [2],
            "marriage": [1],
            "age": [40],
            "default_next_month": [0],
            "ingested_at": [pd.Timestamp("2026-01-01")],
            "bill_amt_1": [500],
            "bill_amt_2": [0],
            "bill_amt_3": [0],
            "bill_amt_4": [0],
            "bill_amt_5": [0],
            "bill_amt_6": [0],
            "pay_amt_1": [0],
            "pay_amt_2": [0],
            "pay_amt_3": [0],
            "pay_amt_4": [0],
            "pay_amt_5": [0],
            "pay_amt_6": [0],
        }
    )
    for i in range(1, 7):
        row[f"pay_status_{i}"] = -2

    from credit_risk.features.build import build_features

    warehouse.write_table(row, settings.data.raw_table, mode="replace")
    features = build_features(settings, warehouse, validate=False)

    assert pd.isna(features["overall_payment_ratio"].iloc[0])
    assert features["has_billing_history"].iloc[0] == 0
    assert features["months_no_consumption"].iloc[0] == 6


def test_delinquency_counters(feature_frame, warehouse, settings):
    """months_delinquent counts pay_status >= 1; revolving (0) is excluded."""
    clean = warehouse.query(f"SELECT * FROM {settings.data.clean_table}").set_index("client_id")
    features = feature_frame.set_index("client_id")
    status_cols = [f"pay_status_{i}" for i in range(1, 7)]

    expected = (clean[status_cols] >= 1).sum(axis=1)
    assert (features.loc[expected.index, "months_delinquent"] == expected).all()

    expected_revolving = (clean[status_cols] == 0).sum(axis=1)
    assert (features.loc[expected.index, "months_revolving"] == expected_revolving).all()


def test_bill_volatility_matches_population_std(feature_frame, warehouse, settings):
    """The E[x^2] - E[x]^2 formulation must equal a population standard deviation."""
    clean = warehouse.query(f"SELECT * FROM {settings.data.clean_table}").set_index("client_id")
    features = feature_frame.set_index("client_id")
    bills = [f"bill_amt_{i}" for i in range(1, 7)]

    expected = clean[bills].std(axis=1, ddof=0)
    actual = features.loc[expected.index, "bill_volatility"]
    assert np.allclose(expected.to_numpy(), actual.to_numpy(), atol=1e-6)


def test_split_is_deterministic_and_disjoint(feature_frame, settings):
    train, valid, test = split_frames(feature_frame, settings)
    assert len(train) + len(valid) + len(test) == len(feature_frame)

    # No client may appear in more than one split - that would leak.
    ids = [set(frame["client_id"]) for frame in (train, valid, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_split_is_stable_across_rebuilds(settings, warehouse, raw_frame):
    """Rebuilding must not move clients between splits."""
    from credit_risk.features.build import build_features

    warehouse.write_table(raw_frame, settings.data.raw_table, mode="replace")
    first = build_features(settings, warehouse)[["client_id", "split_name"]]

    # Rebuild from the same source, as a scheduled rerun would.
    warehouse.write_table(raw_frame, settings.data.raw_table, mode="replace")
    second = build_features(settings, warehouse)[["client_id", "split_name"]]

    merged = first.merge(second, on="client_id", suffixes=("_a", "_b"))
    assert (merged["split_name_a"] == merged["split_name_b"]).all()


def test_split_assignment_ignores_new_clients(settings, warehouse, raw_frame):
    """Adding clients must not reassign existing ones.

    This is the property a hash-based split has and a random shuffle does not:
    without it, yesterday's test client becomes today's training client and the
    held-out set is quietly contaminated.
    """
    from credit_risk.features.build import build_features

    warehouse.write_table(raw_frame, settings.data.raw_table, mode="replace")
    before = build_features(settings, warehouse)[["client_id", "split_name"]]

    extra = raw_frame.copy()
    extra["client_id"] = extra["client_id"] + 10_000
    warehouse.write_table(
        pd.concat([raw_frame, extra], ignore_index=True), settings.data.raw_table, mode="replace"
    )
    after = build_features(settings, warehouse)[["client_id", "split_name"]]

    merged = before.merge(after, on="client_id", suffixes=("_before", "_after"))
    assert len(merged) == len(before)
    assert (merged["split_name_before"] == merged["split_name_after"]).all()


def test_protected_attribute_excluded_from_features(feature_frame, settings):
    names = feature_columns(feature_frame, settings)
    assert "sex" not in names
    # ...but retained on the frame so fairness can be audited.
    assert "sex" in feature_frame.columns


def test_target_and_identifier_excluded_from_features(feature_frame, settings):
    names = feature_columns(feature_frame, settings)
    assert settings.model.target not in names
    assert "client_id" not in names
    assert "split_name" not in names


def test_validation_rejects_degenerate_target(feature_frame, settings):
    broken = feature_frame.copy()
    broken[settings.model.target] = 0
    with pytest.raises(FeatureValidationError, match="degenerate target"):
        validate_features(broken, settings)


def test_validation_rejects_duplicate_clients(feature_frame, settings):
    broken = pd.concat([feature_frame, feature_frame.head(1)], ignore_index=True)
    with pytest.raises(FeatureValidationError, match="duplicate client_id"):
        validate_features(broken, settings)


def test_validation_rejects_empty_table(settings):
    with pytest.raises(FeatureValidationError, match="empty"):
        validate_features(pd.DataFrame(), settings)


def test_clean_stage_deduplicates(settings, warehouse, raw_frame):
    """A replayed load must not double-count clients."""
    from credit_risk.features.build import build_features

    duplicated = pd.concat([raw_frame, raw_frame], ignore_index=True)
    warehouse.write_table(duplicated, settings.data.raw_table, mode="replace")
    features = build_features(settings, warehouse)
    assert len(features) == len(raw_frame)
