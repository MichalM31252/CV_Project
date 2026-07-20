"""Tests for cost-sensitive evaluation.

The cost curve drives the deployed operating point, so it is verified against
brute force rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.config import CostMatrix
from credit_risk.training.evaluate import (
    baseline_costs,
    calibration_table,
    confusion_at_threshold,
    cost_curve,
    evaluate,
    fairness_report,
    optimal_threshold,
)

COST = CostMatrix(false_negative=16000, false_positive=3000, true_positive=1500, true_negative=0)


@pytest.fixture
def scores():
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.22, 600)
    # Scores correlated with the label so the curve has a real optimum.
    p = np.clip(rng.beta(2, 6, 600) + y * 0.25, 0, 1)
    return y, p


def test_confusion_matches_manual_count(scores):
    y, p = scores
    tn, fp, fn, tp = confusion_at_threshold(y, p, 0.3)
    pred = (p >= 0.3).astype(int)
    assert tp == int(((y == 1) & (pred == 1)).sum())
    assert fp == int(((y == 0) & (pred == 1)).sum())
    assert tn + fp + fn + tp == len(y)


def test_cost_curve_matches_brute_force(scores):
    """Every point on the curve must equal the cost of actually applying it."""
    y, p = scores
    thresholds, costs = cost_curve(y, p, COST)
    brute = np.array([COST.expected_cost(*confusion_at_threshold(y, p, t)) for t in thresholds])
    assert np.allclose(brute, costs)


def test_cost_curve_handles_tied_probabilities():
    """Isotonic calibration emits heavy ties; cut points inside a tie run are
    not achievable operating points and must be excluded."""
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    p = np.array([0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8])
    thresholds, costs = cost_curve(y, p, COST)
    brute = np.array([COST.expected_cost(*confusion_at_threshold(y, p, t)) for t in thresholds])
    assert np.allclose(brute, costs)


def test_optimal_threshold_is_the_true_minimum(scores):
    y, p = scores
    threshold, cost = optimal_threshold(y, p, COST)
    # No other achievable operating point may be cheaper.
    grid = np.unique(np.concatenate([p, [0.0, 1.0]]))
    best_elsewhere = min(COST.expected_cost(*confusion_at_threshold(y, p, t)) for t in grid)
    assert cost <= best_elsewhere + 1e-9
    assert COST.expected_cost(*confusion_at_threshold(y, p, threshold)) == pytest.approx(cost)


def test_asymmetric_costs_push_threshold_below_half():
    """With false negatives dearer than false positives, the cost-optimal
    threshold must sit below 0.5 - flagging more aggressively than a naive
    argmax classifier."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.22, 1000)
    p = np.clip(rng.beta(2, 6, 1000) + y * 0.3, 0, 1)
    threshold, _ = optimal_threshold(y, p, COST)
    assert threshold < 0.5


def test_symmetric_costs_move_threshold_up():
    """Raising the false-positive cost must make the model flag less."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.22, 1000)
    p = np.clip(rng.beta(2, 6, 1000) + y * 0.3, 0, 1)
    cheap_fp, _ = optimal_threshold(y, p, COST)
    expensive_fp, _ = optimal_threshold(
        y,
        p,
        CostMatrix(false_negative=16000, false_positive=15000, true_positive=1500, true_negative=0),
    )
    assert expensive_fp > cheap_fp


def test_baselines_are_the_two_model_free_policies(scores):
    y, _ = scores
    baselines = baseline_costs(y, COST)
    assert baselines["intervene_none"] == pytest.approx(y.sum() * COST.false_negative)
    assert baselines["intervene_all"] == pytest.approx(
        y.sum() * COST.true_positive + (len(y) - y.sum()) * COST.false_positive
    )


def test_evaluate_reports_savings_against_best_baseline(scores):
    y, p = scores
    threshold, _ = optimal_threshold(y, p, COST)
    metrics = evaluate(y, p, COST, threshold)
    best = min(metrics["baseline_intervene_none"], metrics["baseline_intervene_all"])
    assert metrics["savings_vs_best_baseline"] == pytest.approx(best - metrics["total_cost"])
    # A cost-optimised threshold cannot be worse than the best fixed policy.
    assert metrics["savings_vs_best_baseline"] >= -1e-9


def test_perfect_model_scores_perfectly():
    y = np.array([0, 0, 1, 1, 0, 1])
    metrics = evaluate(y, y.astype(float), COST, 0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion"]["fn"] == 0
    assert metrics["confusion"]["fp"] == 0


def test_calibration_table_bins_and_sums(scores):
    y, p = scores
    table = calibration_table(y, p, n_bins=5)
    assert len(table) == 5
    assert table["n"].sum() == len(y)


def test_fairness_report_covers_every_group(scores):
    y, p = scores
    groups = np.where(np.arange(len(y)) % 2 == 0, 1, 2)
    report = fairness_report(y, p, groups, 0.3, group_labels={1: "male", 2: "female"})
    assert set(report["group"]) == {"male", "female"}
    assert report["n"].sum() == len(y)
