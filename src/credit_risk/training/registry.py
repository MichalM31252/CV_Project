"""Model registry - versioned artifacts with the lineage needed to trust them.

A ``.joblib`` on disk is not a deployable model. To serve a score you also need to
know which columns went in and in what order, which threshold turns a probability
into a decision, and which library versions produced the artifact. Without those,
"the API returns different numbers than the notebook" becomes unanswerable.

Every save therefore writes an artifact plus a metadata document recording:

* a version stamp, and the git commit if the tree is clean enough to identify one
* the exact ordered feature list
* the operating threshold and the cost matrix that selected it
* validation and test metrics
* row counts per split
* library versions

Artifacts are written under ``versions/<version>/`` and then copied to a stable
``latest`` name. Serving reads ``latest`` by default but can pin a version, so a
rollback is a config change rather than a retrain.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from ..config import Settings
from ..storage import get_blob_store

logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    """Short commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch  # noqa: PLC0415 - optional at registry level

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    return versions


def new_version() -> str:
    """UTC timestamp version id, lexicographically sortable."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def save_model(
    settings: Settings,
    model: Any,
    flavor: str,
    feature_names: list[str],
    threshold: float,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Persist a model plus its metadata. Returns the metadata document."""
    version = version or new_version()
    store = get_blob_store(settings, prefix="models")
    staging = Path(settings.model_dir)
    staging.mkdir(parents=True, exist_ok=True)

    base = f"{settings.serving.model_name}_{flavor}"
    artifact_name = f"{base}.joblib"
    metadata_name = f"{base}_metadata.json"

    metadata: dict[str, Any] = {
        "model_name": settings.serving.model_name,
        "flavor": flavor,
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "threshold": float(threshold),
        "cost_matrix": settings.model.cost.model_dump(),
        "metrics": metrics,
        "libraries": _library_versions(),
        "artifact": artifact_name,
        **(extra or {}),
    }

    local_artifact = staging / artifact_name
    local_metadata = staging / metadata_name
    joblib.dump(model, local_artifact)
    local_metadata.write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",  # keep artifacts byte-identical across platforms
    )

    # Immutable versioned copy first, then the mutable "latest" pointer, so a
    # crash between the two can never leave `latest` referring to nothing.
    store.upload(local_artifact, f"versions/{version}/{artifact_name}")
    store.upload(local_metadata, f"versions/{version}/{metadata_name}")
    store.upload(local_artifact, f"latest/{artifact_name}")
    store.upload(local_metadata, f"latest/{metadata_name}")

    logger.info(
        "model registered",
        extra={"flavor": flavor, "version": version, "threshold": float(threshold)},
    )
    return metadata


def load_model(
    settings: Settings,
    flavor: str | None = None,
    version: str = "latest",
) -> tuple[Any, dict[str, Any]]:
    """Load a model and its metadata. Returns ``(model, metadata)``."""
    flavor = flavor or settings.serving.model_flavor
    store = get_blob_store(settings, prefix="models")

    base = f"{settings.serving.model_name}_{flavor}"
    artifact_name = f"{base}.joblib"
    metadata_name = f"{base}_metadata.json"
    prefix = "latest" if version == "latest" else f"versions/{version}"

    staging = Path(settings.model_dir) / "_loaded"
    staging.mkdir(parents=True, exist_ok=True)

    artifact_path = store.download(f"{prefix}/{artifact_name}", staging / artifact_name)
    metadata_path = store.download(f"{prefix}/{metadata_name}", staging / metadata_name)

    model = joblib.load(artifact_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info(
        "model loaded",
        extra={"flavor": flavor, "version": metadata.get("version"), "source": prefix},
    )
    return model, metadata
