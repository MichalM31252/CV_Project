"""Pipeline orchestrator and CLI.

Runs the stages end to end, or one at a time:

    python -m credit_risk.pipeline --stage all
    python -m credit_risk.pipeline --stage features
    python -m credit_risk.pipeline --stage train --flavor torch

The same entry point drives local runs, CI and the Cloud Run training job; only
``backend`` in the config differs. Keeping orchestration in one module - rather
than in a notebook that has to be executed top to bottom by hand - is what makes
the pipeline reproducible and schedulable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, Settings, get_settings, load_settings
from .features.build import build_features, feature_columns, split_frames
from .ingest.download import download_raw
from .ingest.load_raw import load_raw_table
from .logging_config import configure_from_settings
from .training import evaluate as ev
from .warehouse import get_warehouse

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"


def stage_ingest(settings: Settings) -> int:
    """Download the source data and load it into the warehouse."""
    parquet_path = download_raw(settings)
    with get_warehouse(settings) as warehouse:
        return load_raw_table(settings, warehouse, parquet_path)


def stage_features(settings: Settings) -> pd.DataFrame:
    """Run the SQL cleaning and feature stages."""
    with get_warehouse(settings) as warehouse:
        return build_features(settings, warehouse)


def stage_train(settings: Settings, flavors: list[str]) -> dict[str, Any]:
    """Train the requested model flavours and report the comparison."""
    with get_warehouse(settings) as warehouse:
        features = warehouse.query(
            f"SELECT * FROM {settings.table_ref(settings.data.features_table)}"  # noqa: S608
        )

    train_df, valid_df, test_df = split_frames(features, settings)
    names = feature_columns(features, settings)
    logger.info(
        "training",
        extra={
            "features": len(names),
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
        },
    )

    outcomes: dict[str, Any] = {}
    if "sklearn" in flavors:
        from .training.train_sklearn import train as train_sklearn  # noqa: PLC0415

        outcomes["sklearn"] = train_sklearn(settings, train_df, valid_df, test_df, names)
    if "torch" in flavors:
        from .training.train_torch import train as train_torch  # noqa: PLC0415

        outcomes["torch"] = train_torch(settings, train_df, valid_df, test_df, names)

    write_report(settings, outcomes, test_df)
    return outcomes


def cost_sensitivity(
    settings: Settings, y_true: Any, y_prob: Any, ratios: list[float] | None = None
) -> pd.DataFrame:
    """How the operating point moves as the FN:FP cost ratio changes.

    The cost matrix encodes an assumption about the business. This sweep shows
    what happens if Risk disagrees with it, which is a more useful thing to hand
    a stakeholder than a single number derived from figures they did not choose.
    """
    from .config import CostMatrix  # noqa: PLC0415

    ratios = ratios or [2, 3, 5, 8, 10, 15, 20, 30]
    base = settings.model.cost
    rows = []
    for ratio in ratios:
        cost = CostMatrix(
            false_negative=base.false_positive * ratio,
            false_positive=base.false_positive,
            true_positive=base.true_positive,
            true_negative=base.true_negative,
        )
        threshold, _ = ev.optimal_threshold(y_true, y_prob, cost)
        metrics = ev.evaluate(y_true, y_prob, cost, threshold)
        rows.append(
            {
                "fn_fp_ratio": ratio,
                "threshold": round(threshold, 4),
                "flagged_pct": round(
                    (metrics["confusion"]["tp"] + metrics["confusion"]["fp"]) / metrics["n"] * 100,
                    1,
                ),
                "recall": round(metrics["recall"], 4),
                "precision": round(metrics["precision"], 4),
                "savings_pct": round(metrics["savings_pct"], 1),
            }
        )
    return pd.DataFrame(rows)


def write_report(settings: Settings, outcomes: dict[str, Any], test_df: pd.DataFrame) -> Path:
    """Write a markdown model report plus machine-readable metrics."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Model Report", ""]
    lines.append(f"Backend: `{settings.backend}` | target: `{settings.model.target}`")
    lines.append("")

    lines.append("## Test set performance")
    lines.append("")
    lines.append(
        "| flavour | algorithm | ROC-AUC | PR-AUC | Brier | precision | recall | F1 | cost | savings vs baseline |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for flavor, outcome in outcomes.items():
        m = outcome["test_metrics"]
        algorithm = outcome.get("selected_algorithm", "neural network")
        lines.append(
            f"| {flavor} | {algorithm} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['brier']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['total_cost']:,.0f} | {m['savings_pct']:.1f}% |"
        )
    lines.append("")

    for flavor, outcome in outcomes.items():
        m = outcome["test_metrics"]
        c = m["confusion"]
        lines.append(f"### {flavor}")
        lines.append("")
        lines.append(
            f"- operating threshold: **{m['threshold']:.4f}** (chosen on validation, cost-minimising)"
        )
        lines.append(
            f"- confusion matrix: TN={c['tn']:,} FP={c['fp']:,} FN={c['fn']:,} TP={c['tp']:,}"
        )
        lines.append(f"- cost per client: NT${m['cost_per_client']:,.0f}")
        lines.append(
            f"- baselines: intervene-nobody NT${m['baseline_intervene_none']:,.0f}, "
            f"intervene-everybody NT${m['baseline_intervene_all']:,.0f}"
        )
        lines.append("")

    if "sklearn" in outcomes:
        best = outcomes["sklearn"]
        if best.get("fairness") is not None:
            lines.append("## Fairness audit (protected attribute excluded from features)")
            lines.append("")
            lines.append(best["fairness"].round(4).to_markdown(index=False))
            lines.append("")

        lines.append("## Calibration (test, by decile)")
        lines.append("")
        lines.append(best["calibration"].round(4).to_markdown(index=False))
        lines.append("")

        lines.append("## Cost sensitivity")
        lines.append("")
        lines.append("Operating point as the false-negative : false-positive cost ratio varies.")
        lines.append("")
        sensitivity = cost_sensitivity(
            settings, test_df[settings.model.target].to_numpy(), best["test_prob"]
        )
        lines.append(sensitivity.to_markdown(index=False))
        lines.append("")

    report_path = REPORTS_DIR / "model_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {flavor: outcome["test_metrics"] for flavor, outcome in outcomes.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("report written", extra={"path": str(report_path)})
    return report_path


def stage_drift(settings: Settings) -> pd.DataFrame:
    """Compare served predictions against the training baseline."""
    from .monitoring.drift import run_drift_check  # noqa: PLC0415

    with get_warehouse(settings) as warehouse:
        return run_drift_check(settings, warehouse)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Credit default risk pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "features", "train", "drift"],
        default="all",
    )
    parser.add_argument(
        "--flavor",
        choices=["sklearn", "torch", "both"],
        default="both",
        help="which model implementation(s) to train",
    )
    parser.add_argument("--config", default=None, help="path to an alternative config file")
    args = parser.parse_args(argv)

    settings = get_settings() if args.config is None else load_settings(args.config)
    configure_from_settings(settings)

    flavors = ["sklearn", "torch"] if args.flavor == "both" else [args.flavor]

    if args.stage in ("all", "ingest"):
        stage_ingest(settings)
    if args.stage in ("all", "features"):
        stage_features(settings)
    if args.stage in ("all", "train"):
        outcomes = stage_train(settings, flavors)
        for flavor, outcome in outcomes.items():
            m = outcome["test_metrics"]
            logger.info(
                "result",
                extra={
                    "flavor": flavor,
                    "pr_auc": round(m["pr_auc"], 4),
                    "roc_auc": round(m["roc_auc"], 4),
                    "savings_pct": round(m["savings_pct"], 1),
                },
            )
    if args.stage == "drift":
        stage_drift(settings)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
