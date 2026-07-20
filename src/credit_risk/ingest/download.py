"""Stage 1 - acquire the source dataset and land it as Parquet.

Source: UCI ML Repository #350, "Default of Credit Card Clients" (Yeh & Lien,
2009). 30,000 Taiwanese credit-card accounts observed April-September 2005, with
a binary label for default in October 2005.

Two things here are deliberate rather than incidental:

*Checksum verification.* The upstream file is fetched over the network from a
third party. Verifying a known SHA-256 means an upstream change becomes a loud
failure at ingestion instead of a quiet distribution shift that only shows up as
degraded model metrics weeks later.

*Column renaming.* The raw file carries the well-known ``PAY_0`` anomaly: the
repayment-status columns run ``PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6`` with no
``PAY_1``. ``PAY_0`` is in fact the most recent month. Left alone this misleads
anyone reading the feature code into thinking a month is missing. We normalise to
``pay_status_1..6`` where index 1 is the most recent month (September 2005) and
index 6 the oldest (April 2005), consistent with ``bill_amt_*`` and ``pay_amt_*``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import Settings

logger = logging.getLogger(__name__)

# Raw UCI header -> canonical snake_case name.
COLUMN_RENAMES: dict[str, str] = {
    "ID": "client_id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    # Repayment status. PAY_0 is the most recent month despite the name.
    "PAY_0": "pay_status_1",
    "PAY_2": "pay_status_2",
    "PAY_3": "pay_status_3",
    "PAY_4": "pay_status_4",
    "PAY_5": "pay_status_5",
    "PAY_6": "pay_status_6",
    # Bill statement amounts, index 1 = most recent.
    **{f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)},
    # Amount of previous payment, index 1 = most recent.
    **{f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)},
    "default payment next month": "default_next_month",
}

EXPECTED_ROWS = 30_000


def _fetch_bytes(url: str, timeout: int = 120) -> bytes:
    logger.info("downloading source dataset", extra={"url": url})
    request = urllib.request.Request(url, headers={"User-Agent": "credit-risk-pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL from config
        return response.read()


def _extract_member(archive: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        if member not in names:
            raise FileNotFoundError(f"{member!r} not in archive; found {names}")
        return zf.read(member)


def _verify_checksum(payload: bytes, expected_sha256: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            "source data checksum mismatch - upstream file has changed.\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            "Review the upstream dataset, then update data.source_sha256 in "
            "config/config.yaml once the change is understood."
        )
    logger.info("source checksum verified", extra={"sha256": actual})


def download_raw(settings: Settings, force: bool = False) -> Path:
    """Download, verify and convert the source dataset to Parquet.

    Returns the path to the local Parquet file. Re-downloading is skipped when the
    output already exists unless ``force`` is set.
    """
    raw_dir = Path(settings.local.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "credit_default_raw.parquet"

    if parquet_path.exists() and not force:
        logger.info(
            "raw parquet already present, skipping download", extra={"path": str(parquet_path)}
        )
        return parquet_path

    archive = _fetch_bytes(settings.data.source_url)
    payload = _extract_member(archive, settings.data.source_member)
    _verify_checksum(payload, settings.data.source_sha256)

    # Row 0 of the sheet is a group banner ("X1, X2, ..."); the real header is row 1.
    df = pd.read_excel(io.BytesIO(payload), header=1)

    missing = set(COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"source schema changed; missing expected columns: {sorted(missing)}")

    df = df.rename(columns=COLUMN_RENAMES)[list(COLUMN_RENAMES.values())]

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")

    # Lineage columns. In a warehouse the raw layer should record where each row
    # came from and when, so a bad load can be identified and reverted.
    df["ingested_at"] = datetime.now(UTC).replace(tzinfo=None)
    df["source_uri"] = settings.data.source_url

    df.to_parquet(parquet_path, index=False)
    logger.info(
        "raw dataset written",
        extra={"path": str(parquet_path), "rows": len(df), "columns": len(df.columns)},
    )
    return parquet_path
