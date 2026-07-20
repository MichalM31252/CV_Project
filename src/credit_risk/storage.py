"""Blob storage abstraction for model artifacts.

Mirrors the warehouse split: Cloud Storage in production, a local directory when
running offline. Training writes artifacts through this interface and serving
reads them back, so promoting a model from a laptop to Cloud Run is a config
change rather than a code change.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)


class BlobStore(ABC):
    @abstractmethod
    def upload(self, local_path: Path, remote_path: str) -> str:
        """Store a local file; returns a URI identifying the stored object."""

    @abstractmethod
    def download(self, remote_path: str, local_path: Path) -> Path:
        """Fetch an object to a local path; returns that path."""

    @abstractmethod
    def exists(self, remote_path: str) -> bool: ...


class LocalBlobStore(BlobStore):
    """Filesystem-backed store rooted at a base directory."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, remote_path: str) -> Path:
        return self.base_dir / remote_path

    def upload(self, local_path: Path, remote_path: str) -> str:
        target = self._resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Skip the copy when training already wrote straight to the target.
        if Path(local_path).resolve() != target.resolve():
            shutil.copy2(local_path, target)
        return str(target)

    def download(self, remote_path: str, local_path: Path) -> Path:
        source = self._resolve(remote_path)
        if not source.exists():
            raise FileNotFoundError(f"artifact not found: {source}")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != local_path.resolve():
            shutil.copy2(source, local_path)
        return local_path

    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).exists()


class GCSBlobStore(BlobStore):
    """Google Cloud Storage-backed store."""

    def __init__(self, bucket_name: str, prefix: str = "") -> None:
        from google.cloud import storage  # noqa: PLC0415 - deliberate lazy import

        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def _blob_name(self, remote_path: str) -> str:
        return f"{self.prefix}/{remote_path}" if self.prefix else remote_path

    def upload(self, local_path: Path, remote_path: str) -> str:
        blob = self.bucket.blob(self._blob_name(remote_path))
        blob.upload_from_filename(str(local_path))
        uri = f"gs://{self.bucket_name}/{self._blob_name(remote_path)}"
        logger.info("uploaded artifact", extra={"uri": uri})
        return uri

    def download(self, remote_path: str, local_path: Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(self._blob_name(remote_path)).download_to_filename(str(local_path))
        return local_path

    def exists(self, remote_path: str) -> bool:
        return self.bucket.blob(self._blob_name(remote_path)).exists()


def get_blob_store(settings: Settings, prefix: str = "models") -> BlobStore:
    """Build the artifact store for the configured backend."""
    if settings.backend == "gcp":
        return GCSBlobStore(settings.gcp.artifact_bucket, prefix=prefix)
    return LocalBlobStore(settings.local.model_dir)
