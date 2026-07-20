"""Stage 4 - train, calibrate and select the scikit-learn model.

Design notes
------------
*Two candidates, not one.* A regularised logistic regression is trained
alongside gradient boosting. The linear model is not decoration: it is the
honest baseline that says how much the extra complexity is actually buying, and
in credit risk a model you can explain coefficient-by-coefficient has real
regulatory value. The winner is chosen on validation PR-AUC.

*Three splits, each with one job.* Hyperparameters are searched with
cross-validation inside **train**. Calibration and the decision threshold are
fitted on **valid**. **Test** is touched exactly once, at the end. Selecting a
threshold on test is the most common way portfolio projects quietly overstate
their results.

*Calibration before thresholding.* Gradient boosting optimises log-loss but its
raw scores are not well-calibrated, and class weighting distorts them further. A
cost-derived threshold is only meaningful on calibrated probabilities, so the
selected model is wrapped in isotonic calibration fitted on validation data
before the threshold is chosen.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import Settings
from . import evaluate as ev
from .registry import save_model

logger = logging.getLogger(__name__)

# Nominal categories - the integer codes carry no order, so they are one-hot
# encoded. Everything else (including the ordinal pay_status_* scale, where
# larger really does mean worse) is treated as numeric.
CATEGORICAL_FEATURES = ["education", "marriage"]


def _build_preprocessor(feature_names: list[str], scale: bool, impute: bool) -> ColumnTransformer:
    """One-hot the nominal columns; optionally impute and scale the rest.

    Scaling matters for logistic regression (L2 penalises large coefficients, so
    unscaled features are penalised unequally) and is irrelevant for trees, which
    only compare split points.

    Imputation is applied only where the estimator demands it. ``overall_payment_ratio``
    is NULL for dormant accounts, which is real information rather than a data
    quality defect - and those accounts default at a materially higher rate.
    Gradient boosting routes NaN down its own branch and learns that directly, so
    it gets the raw values. Logistic regression cannot accept NaN, so it gets a
    median fill, with ``has_billing_history`` carrying the signal that the fill
    would otherwise erase.
    """
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    numeric = [c for c in feature_names if c not in categorical]

    if impute and scale:
        numeric_transformer: Any = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
    elif impute:
        numeric_transformer = SimpleImputer(strategy="median")
    elif scale:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                # Unknown categories at serving time must not crash the API.
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                categorical,
            ),
            ("numeric", numeric_transformer, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidate_models(settings: Settings, feature_names: list[str]) -> dict[str, dict[str, Any]]:
    """Candidate estimators with their search spaces."""
    seed = settings.model.random_state
    return {
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("preprocess", _build_preprocessor(feature_names, scale=True, impute=True)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            # Re-weight the 22/78 split so the minority class is
                            # not simply ignored by the loss.
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            # Only C is tuned: `penalty` is deprecated in scikit-learn 1.8 in
            # favour of `l1_ratio`, and the default (L2) is what we want here.
            "search_space": {"model__C": np.logspace(-3, 2, 20)},
        },
        "hist_gradient_boosting": {
            "pipeline": Pipeline(
                [
                    # No imputation: HistGradientBoosting handles NaN natively and
                    # learns the dormant-account split for itself.
                    ("preprocess", _build_preprocessor(feature_names, scale=False, impute=False)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            class_weight="balanced",
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=20,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 4, 6, 8, None],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [20, 50, 100, 200],
                "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
                "model__max_iter": [200, 400],
            },
        },
    }


def train(
    settings: Settings,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Train, calibrate, threshold and register the scikit-learn model."""
    target = settings.model.target
    cost = settings.model.cost

    x_train, y_train = train_df[feature_names], train_df[target].to_numpy()
    x_valid, y_valid = valid_df[feature_names], valid_df[target].to_numpy()
    x_test, y_test = test_df[feature_names], test_df[target].to_numpy()

    logger.info(
        "training scikit-learn candidates",
        extra={"train": len(x_train), "valid": len(x_valid), "test": len(x_test)},
    )

    cv = StratifiedKFold(
        n_splits=settings.model.sklearn.cv_folds,
        shuffle=True,
        random_state=settings.model.random_state,
    )

    results: dict[str, dict[str, Any]] = {}
    for name, spec in _candidate_models(settings, feature_names).items():
        search = RandomizedSearchCV(
            spec["pipeline"],
            spec["search_space"],
            n_iter=settings.model.sklearn.n_iter_search,
            # Average precision, not accuracy: the positive class is what matters
            # and it is the minority.
            scoring="average_precision",
            cv=cv,
            random_state=settings.model.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)

        valid_prob = search.best_estimator_.predict_proba(x_valid)[:, 1]
        valid_pr_auc = float(ev.average_precision_score(y_valid, valid_prob))
        results[name] = {
            "estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_pr_auc": float(search.best_score_),
            "valid_pr_auc": valid_pr_auc,
        }
        logger.info(
            "candidate trained",
            extra={
                "candidate": name,
                "cv_pr_auc": float(search.best_score_),
                "valid_pr_auc": valid_pr_auc,
            },
        )

    # Select on validation, not on the cross-validation score: the CV score was
    # what the search optimised, so it is optimistically biased.
    best_name = max(results, key=lambda n: results[n]["valid_pr_auc"])
    best = results[best_name]
    logger.info("selected candidate", extra={"candidate": best_name})

    # Calibrate the frozen winner on validation data. FrozenEstimator prevents
    # CalibratedClassifierCV from refitting the base model on the calibration set,
    # which would leak validation data into the fit.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(best["estimator"]),
        method=settings.model.sklearn.calibration_method,
    )
    calibrated.fit(x_valid, y_valid)

    # Threshold from the calibrated scores on validation.
    valid_prob_cal = calibrated.predict_proba(x_valid)[:, 1]
    threshold, valid_cost = ev.optimal_threshold(y_valid, valid_prob_cal, cost)

    uncalibrated_valid = best["estimator"].predict_proba(x_valid)[:, 1]
    logger.info(
        "threshold selected on validation",
        extra={
            "threshold": threshold,
            "valid_cost": valid_cost,
            "brier_before_calibration": float(ev.brier_score_loss(y_valid, uncalibrated_valid)),
            "brier_after_calibration": float(ev.brier_score_loss(y_valid, valid_prob_cal)),
        },
    )

    # Single, final look at the held-out test set.
    test_prob = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = ev.evaluate(y_test, test_prob, cost, threshold)
    valid_metrics = ev.evaluate(y_valid, valid_prob_cal, cost, threshold)

    calibration = ev.calibration_table(y_test, test_prob)

    # Fairness audit on the protected attribute, which is deliberately absent
    # from the feature set but still present in the frame.
    fairness = None
    if "sex" in test_df.columns:
        fairness = ev.fairness_report(
            y_test, test_prob, test_df["sex"], threshold, group_labels={1: "male", 2: "female"}
        )

    metadata = save_model(
        settings,
        calibrated,
        flavor="sklearn",
        feature_names=feature_names,
        threshold=threshold,
        metrics={"valid": valid_metrics, "test": test_metrics},
        extra={
            "selected_algorithm": best_name,
            "best_params": {k: str(v) for k, v in best["best_params"].items()},
            "cv_pr_auc": best["cv_pr_auc"],
            "calibration_method": settings.model.sklearn.calibration_method,
            "candidate_scores": {n: r["valid_pr_auc"] for n, r in results.items()},
            "split_sizes": {"train": len(x_train), "valid": len(x_valid), "test": len(x_test)},
        },
    )

    return {
        "model": calibrated,
        "metadata": metadata,
        "test_metrics": test_metrics,
        "valid_metrics": valid_metrics,
        "calibration": calibration,
        "fairness": fairness,
        "selected_algorithm": best_name,
        "candidates": {n: r["valid_pr_auc"] for n, r in results.items()},
        "test_prob": test_prob,
    }
