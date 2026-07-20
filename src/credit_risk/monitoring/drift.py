"""Data drift detection via Population Stability Index.

A deployed model degrades long before anyone has labels to prove it. Defaults are
only observable a month later, so waiting for ground truth means running blind
for a full cycle. Input drift is observable immediately, which makes it the
practical early warning.

PSI compares the distribution of a feature in production against the training
baseline:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Conventional reading, which ``config.yaml`` follows:

    < 0.10   stable
    0.10-0.25  moderate shift, investigate
    > 0.25   significant shift, retraining likely needed

Two implementation details that matter:

*Bin edges come from the training baseline*, not from the production sample.
Re-binning on production data would rescale both distributions together and hide
the very shift being looked for.

*Empty bins are floored* at a small epsilon. A production bin with zero mass
gives ln(0) = -inf, which would report infinite drift for what may be a handful
of missing rows.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..warehouse import Warehouse

logger = logging.getLogger(__name__)

EPSILON = 1e-6
DEFAULT_BINS = 10


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = DEFAULT_BINS,
) -> tuple[float, pd.DataFrame]:
    """PSI of ``actual`` against the ``expected`` baseline.

    Returns the index and a per-bin breakdown, which is what makes a PSI alert
    actionable - it shows *where* the mass moved.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan"), pd.DataFrame()

    # Quantile edges from the baseline. Deduplicated because heavily tied
    # features (counts, flags) produce repeated quantiles that would create
    # zero-width bins.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        # Constant baseline: drift is only meaningful as "did it stop being
        # constant", which a PSI of 0 correctly represents when it did not.
        return 0.0, pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = np.maximum(expected_counts / len(expected), EPSILON)
    actual_pct = np.maximum(actual_counts / len(actual), EPSILON)

    contributions = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    breakdown = pd.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "expected_pct": expected_pct,
            "actual_pct": actual_pct,
            "psi_contribution": contributions,
        }
    )
    return float(np.sum(contributions)), breakdown


def classify(psi: float, settings: Settings) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi >= settings.monitoring.psi_alert:
        return "alert"
    if psi >= settings.monitoring.psi_warn:
        return "warn"
    return "stable"


def compute_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    settings: Settings,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """PSI for every tracked feature present in both frames."""
    features = features or settings.monitoring.tracked_features
    rows: list[dict[str, Any]] = []
    computed_at = datetime.now(UTC).replace(tzinfo=None)

    for feature in features:
        if feature not in baseline.columns or feature not in current.columns:
            logger.warning("tracked feature missing, skipping", extra={"feature": feature})
            continue
        psi, _ = population_stability_index(
            baseline[feature].to_numpy(), current[feature].to_numpy()
        )
        rows.append(
            {
                "computed_at": computed_at,
                "feature": feature,
                "psi": psi,
                "status": classify(psi, settings),
                "baseline_mean": float(np.nanmean(baseline[feature].to_numpy(dtype=float))),
                "current_mean": float(np.nanmean(current[feature].to_numpy(dtype=float))),
                "baseline_n": int(len(baseline)),
                "current_n": int(len(current)),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        alerts = result[result["status"] == "alert"]["feature"].tolist()
        if alerts:
            # ERROR severity so a Cloud Logging alerting policy can page on it.
            logger.error("drift alert", extra={"features": alerts})
        else:
            logger.info("drift check complete", extra={"max_psi": float(result["psi"].max())})
    return result


def run_drift_check(
    settings: Settings,
    warehouse: Warehouse,
    persist: bool = True,
) -> pd.DataFrame:
    """Compare recent served predictions against the training baseline.

    Reads the training split as the baseline and the prediction log as current
    traffic, so this can run as a scheduled Cloud Run job against live data.
    """
    features_table = settings.table_ref(settings.data.features_table)
    predictions_table = settings.table_ref(settings.data.predictions_table)

    baseline = warehouse.query(
        f"SELECT * FROM {features_table} WHERE split_name = 'train'"  # noqa: S608 - identifiers from config
    )

    if not warehouse.table_exists(predictions_table):
        logger.warning("no prediction log yet, drift check skipped")
        return pd.DataFrame()

    current = warehouse.query(f"SELECT * FROM {predictions_table}")  # noqa: S608
    if current.empty:
        logger.warning("prediction log is empty, drift check skipped")
        return pd.DataFrame()

    # Only the raw inputs are echoed into the prediction log, so drift is
    # measured on the columns the two tables genuinely share.
    shared = [c for c in settings.monitoring.tracked_features if c in current.columns]
    result = compute_drift(baseline, current, settings, features=shared)

    if persist and not result.empty:
        warehouse.write_table(result, settings.table_ref(settings.data.drift_table), mode="append")
    return result
