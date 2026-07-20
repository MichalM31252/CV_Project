"""Cost-sensitive evaluation.

Why this module exists rather than a call to ``accuracy_score``:

22% of clients in this dataset default. A model that predicts "no default" for
everybody scores 78% accuracy and is worth nothing. Worse, the two error types
are not equally expensive - missing a default writes off an outstanding balance,
while a false alarm costs an unnecessary intervention. With the cost matrix in
``config.yaml`` a missed default costs roughly 13x a false alarm.

So the pipeline optimises what the business actually pays:

* ranking quality is measured with **PR-AUC** (average precision), which is
  sensitive to performance on the rare positive class in a way ROC-AUC is not;
* probability quality is measured with the **Brier score**, because the decision
  threshold is only meaningful if the scores are calibrated;
* the operating point is chosen by **minimising expected cost** on validation
  data, never on test;
* results are reported against the two decisions available without a model -
  intervene with nobody, or intervene with everybody - so "is this model worth
  deploying" has a number attached.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import CostMatrix


def confusion_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for ``y_prob >= threshold``."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def cost_curve(
    y_true: np.ndarray, y_prob: np.ndarray, cost: CostMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost across every distinct operating point.

    Evaluates the exact set of thresholds that change a decision (each distinct
    predicted probability) instead of an arbitrary grid, so the reported optimum
    is the true optimum rather than the best point that happened to be sampled.

    Returns ``(thresholds, costs)`` ordered by descending threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    order = np.argsort(-y_prob, kind="mergesort")  # stable: ties keep input order
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    total_pos = int(y_true.sum())
    total_neg = int(len(y_true) - total_pos)

    # k = number of clients flagged (the top-k by score).
    tp_cum = np.concatenate([[0], np.cumsum(y_sorted)])
    fp_cum = np.concatenate([[0], np.cumsum(1 - y_sorted)])

    tp = tp_cum
    fp = fp_cum
    fn = total_pos - tp
    tn = total_neg - fp

    costs = (
        tn * cost.true_negative
        + fp * cost.false_positive
        + fn * cost.false_negative
        + tp * cost.true_positive
    )

    # Threshold that yields each k: just above the top score flags nobody;
    # thereafter flag everything scoring >= the k-th highest probability.
    thresholds = np.concatenate([[np.nextafter(p_sorted[0], np.inf)], p_sorted])

    # Drop cut points that fall inside a run of tied probabilities. Thresholding
    # at a tied value flags the whole run, so "top k" is not achievable there and
    # the cost at that index would describe a decision the threshold cannot make.
    # Isotonic calibration emits a step function with many ties, so without this
    # the reported optimum can be an operating point that does not exist.
    keep = np.ones(len(thresholds), dtype=bool)
    if len(p_sorted) > 1:
        keep[1:-1] = p_sorted[1:] < p_sorted[:-1]
    return thresholds[keep], costs[keep]


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, cost: CostMatrix
) -> tuple[float, float]:
    """Cost-minimising threshold and its expected cost.

    Must be fitted on validation data. Choosing it on test would leak the test
    set into the decision and overstate deployed performance.
    """
    thresholds, costs = cost_curve(y_true, y_prob, cost)
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def baseline_costs(y_true: np.ndarray, cost: CostMatrix) -> dict[str, float]:
    """Cost of the two model-free policies, for comparison."""
    y_true = np.asarray(y_true).astype(int)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    return {
        # Do nothing: every default is missed.
        "intervene_none": pos * cost.false_negative + neg * cost.true_negative,
        # Blanket intervention: every non-defaulter is a false alarm.
        "intervene_all": pos * cost.true_positive + neg * cost.false_positive,
    }


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost: CostMatrix,
    threshold: float,
) -> dict[str, Any]:
    """Full metric set at a fixed operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_at_threshold(y_true, y_prob, threshold)
    total_cost = cost.expected_cost(tn=tn, fp=fp, fn=fn, tp=tp)
    baselines = baseline_costs(y_true, cost)
    best_baseline = min(baselines.values())
    n = len(y_true)

    return {
        "n": n,
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        # Threshold-independent ranking and calibration quality.
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        # Point metrics at the chosen operating point.
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        # What the business pays.
        "total_cost": float(total_cost),
        "cost_per_client": float(total_cost / n),
        "baseline_intervene_none": float(baselines["intervene_none"]),
        "baseline_intervene_all": float(baselines["intervene_all"]),
        "savings_vs_best_baseline": float(best_baseline - total_cost),
        "savings_pct": float((best_baseline - total_cost) / best_baseline * 100)
        if best_baseline
        else 0.0,
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by probability decile.

    A well-calibrated model can have its scores read as probabilities, which is
    what makes the cost-based threshold defensible to a risk team.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    # Rank-based bins keep deciles populated despite the skewed score distribution.
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    out["gap"] = out["mean_predicted"] - out["observed_rate"]
    return out


def fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    threshold: float,
    group_labels: dict[Any, str] | None = None,
) -> pd.DataFrame:
    """Per-group performance on a protected attribute.

    The attribute is excluded from the feature set, but exclusion alone does not
    guarantee equal treatment - correlated features can reproduce the same
    disparity. Measuring selection rate and recall per group is the check that
    the exclusion actually achieved something.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    for value in sorted(pd.Series(groups).unique()):
        mask = np.asarray(groups == value)
        if mask.sum() == 0:
            continue
        gt, gp, gpred = y_true[mask], y_prob[mask], y_pred[mask]
        rows.append(
            {
                "group": group_labels.get(value, str(value)) if group_labels else str(value),
                "n": int(mask.sum()),
                "actual_default_rate": float(gt.mean()),
                # Fraction flagged - the disparate-impact surface.
                "selection_rate": float(gpred.mean()),
                "recall": float(recall_score(gt, gpred, zero_division=0)),
                "precision": float(precision_score(gt, gpred, zero_division=0)),
                "roc_auc": float(roc_auc_score(gt, gp)) if len(set(gt)) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)
