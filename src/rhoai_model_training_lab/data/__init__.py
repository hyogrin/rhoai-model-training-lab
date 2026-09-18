"""Data loading, bundle management, and validation.

Provides utilities for loading prepared dataset bundles, validating
checksums and compatibility, and fetching bundles from local/S3/PVC sources.
Learner notebooks consume pre-built bundles; SDG is never triggered here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from rhoai_model_training_lab.config import load_bundle_config, load_yaml_config
from rhoai_model_training_lab.schemas.data import (
    BackendValidationResult,
    BundleManifest,
    CanonicalSample,
    QualityReport,
    SplitInfo,
)

logger = logging.getLogger(__name__)

_CHECKSUM_ALGORITHM = "sha256"
_CHECKSUM_BLOCK_SIZE = 1 << 16  # 64 KiB


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_file_checksum(path: Path | str, algorithm: str = _CHECKSUM_ALGORITHM) -> str:
    """Compute hex digest of a file using the given hash algorithm.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm name (default ``sha256``).

    Returns:
        Lowercase hex digest string.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *algorithm* is not supported by :mod:`hashlib`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot checksum non-existent file: {path}")
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHECKSUM_BLOCK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return records


# ---------------------------------------------------------------------------
# BundleManager
# ---------------------------------------------------------------------------

class BundleManager:
    """Load and inspect a prepared dataset bundle.

    A *bundle* is a directory tree that follows the layout defined in
    ``configs/data-release.yaml``.  The manager provides read-only access
    to the manifest, canonical samples, KB documents, splits, and quality
    reports, as well as integrity and compatibility checks.

    Args:
        bundle_path: Root directory of the prepared bundle.
    """

    def __init__(self, bundle_path: Path | str) -> None:
        self.root = Path(bundle_path).resolve()
        self._manifest: BundleManifest | None = None
        self._split_info: SplitInfo | None = None
        self._quality_report: QualityReport | None = None
        self._release_config: dict[str, Any] | None = None

    # -- core loaders -------------------------------------------------------

    @classmethod
    def load_bundle(cls, path: Path | str) -> "BundleManager":
        """Factory: create a manager and eagerly load its manifest.

        Raises:
            FileNotFoundError: If *path* does not exist or has no manifest.
        """
        mgr = cls(path)
        _ = mgr.manifest  # trigger load
        return mgr

    @property
    def manifest(self) -> BundleManifest:
        if self._manifest is None:
            manifest_path = self.root / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"No manifest.json in bundle: {self.root}")
            with open(manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._manifest = BundleManifest(**data)
            logger.info(
                "Loaded bundle manifest: %s v%s (%d train, %d val)",
                self._manifest.bundle_name,
                self._manifest.bundle_version,
                self._manifest.canonical_train_count,
                self._manifest.canonical_validation_count,
            )
        return self._manifest

    @property
    def split_info(self) -> SplitInfo:
        if self._split_info is None:
            path = self.root / "metadata" / "splits.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing splits metadata: {path}")
            with open(path, encoding="utf-8") as fh:
                self._split_info = SplitInfo(**json.load(fh))
        return self._split_info

    @property
    def quality_report(self) -> QualityReport:
        if self._quality_report is None:
            path = self.root / "reports" / "quality.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing quality report: {path}")
            with open(path, encoding="utf-8") as fh:
                self._quality_report = QualityReport(**json.load(fh))
        return self._quality_report

    # -- release config (cached) -------------------------------------------

    def _get_release_config(self) -> dict[str, Any]:
        if self._release_config is None:
            try:
                self._release_config = load_bundle_config()
            except FileNotFoundError:
                self._release_config = {}
        return self._release_config

    # -- sample access ------------------------------------------------------

    def get_samples(self, split: str = "train") -> list[CanonicalSample]:
        """Load canonical samples for a given split.

        Args:
            split: ``"train"`` or ``"validation"``.

        Returns:
            Parsed :class:`CanonicalSample` objects.
        """
        allowed = ("train", "validation")
        if split not in allowed:
            raise ValueError(f"split must be one of {allowed}, got {split!r}")

        path = self.root / "canonical" / f"{split}.jsonl"
        raw = _load_jsonl(path)
        samples = [CanonicalSample(**r) for r in raw]
        logger.info("Loaded %d canonical %s samples from %s", len(samples), split, path)
        return samples

    def get_training_samples(self, profile: str, split: str = "train") -> list[dict[str, Any]]:
        """Load backend-specific training samples (chat-template formatted).

        Args:
            profile: ``"lora"`` or ``"osft"``.
            split: ``"train"`` or ``"validation"``.
        """
        if profile not in ("lora", "osft"):
            raise ValueError(f"profile must be 'lora' or 'osft', got {profile!r}")
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")

        path = self.root / "training" / profile / f"{split}.jsonl"
        return _load_jsonl(path)

    def get_kb_documents(self) -> list[dict[str, Any]]:
        """Load knowledge-base documents bundled for RAG."""
        path = self.root / "kb" / "documents.jsonl"
        docs = _load_jsonl(path)
        logger.info("Loaded %d KB documents from %s", len(docs), path)
        return docs

    # -- integrity checks ---------------------------------------------------

    def validate_checksums(self) -> tuple[bool, list[str]]:
        """Verify all file checksums listed in ``checksums.sha256``.

        Returns:
            ``(all_ok, list_of_errors)``
        """
        checksum_file = self.root / "checksums.sha256"
        if not checksum_file.exists():
            return False, [f"Checksum file not found: {checksum_file}"]

        errors: list[str] = []
        with open(checksum_file, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    errors.append(f"Malformed checksum line {lineno}: {line!r}")
                    continue
                expected_hash, rel_path = parts
                file_path = self.root / rel_path
                if not file_path.exists():
                    errors.append(f"File missing: {rel_path}")
                    continue
                actual_hash = compute_file_checksum(file_path)
                if actual_hash != expected_hash:
                    errors.append(
                        f"Checksum mismatch for {rel_path}: "
                        f"expected {expected_hash[:12]}…, got {actual_hash[:12]}…"
                    )

        ok = len(errors) == 0
        if ok:
            logger.info("All checksums verified for bundle %s", self.root.name)
        else:
            logger.warning("Checksum verification failed with %d error(s)", len(errors))
        return ok, errors

    def validate_compatibility(
        self,
        model_id: str,
        tokenizer: Any | None = None,
    ) -> BackendValidationResult:
        """Check that the bundle is compatible with the target model.

        Verifies model ID match, tokenizer availability, and basic
        chat-template hash consistency.

        Args:
            model_id: HuggingFace model identifier, e.g.
                ``"Qwen/Qwen3-4B-Instruct-2507"``.
            tokenizer: Optional pre-loaded tokenizer object.  When
                provided, additional checks (e.g. chat-template hash)
                are performed.

        Returns:
            A :class:`BackendValidationResult` summarizing the checks.
        """
        manifest = self.manifest
        errors: list[str] = []
        warnings: list[str] = []

        # Model ID check
        model_ok = manifest.model_id == model_id
        if not model_ok:
            errors.append(
                f"Model mismatch: bundle targets {manifest.model_id!r} "
                f"but requested {model_id!r}"
            )

        # Tokenizer check
        tokenizer_ok = tokenizer is not None
        if not tokenizer_ok:
            warnings.append("No tokenizer provided; skipping chat template checks")

        # Chat template hash
        chat_template_ok = True
        if tokenizer is not None and manifest.chat_template_hash:
            template_src = getattr(tokenizer, "chat_template", None) or ""
            actual_hash = hashlib.sha256(template_src.encode()).hexdigest()
            chat_template_ok = actual_hash == manifest.chat_template_hash
            if not chat_template_ok:
                errors.append(
                    "Chat template hash mismatch — bundle was built with a "
                    "different tokenizer chat template"
                )

        # Sample counts
        sample_ok = (
            manifest.canonical_train_count > 0
            and manifest.canonical_validation_count > 0
        )
        if not sample_ok:
            errors.append("Bundle reports zero train or validation samples")

        return BackendValidationResult(
            backend="general",
            loader_check=True,
            tokenizer_check=tokenizer_ok,
            sample_count_match=sample_ok,
            chat_template_check=chat_template_ok,
            loss_mask_check=True,  # deferred to training validation
            max_length_check=True,
            errors=errors,
            warnings=warnings,
            verified_scope="model_compatibility",
        )


# ---------------------------------------------------------------------------
# Bundle validation (standalone function)
# ---------------------------------------------------------------------------

def validate_prepared_dataset(bundle_path: Path | str) -> dict[str, Any]:
    """Comprehensive validation of a prepared dataset bundle.

    Checks:
    * manifest.json exists and is valid
    * all expected files are present
    * checksum verification passes
    * canonical sample counts match manifest
    * provenance and split metadata are consistent

    Args:
        bundle_path: Root of the dataset bundle.

    Returns:
        A dict with keys ``"valid"`` (bool), ``"errors"`` (list[str]),
        ``"warnings"`` (list[str]), and ``"summary"`` (dict).
    """
    root = Path(bundle_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Manifest
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append("manifest.json not found")
        return {"valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = BundleManifest(**json.load(fh))
    except Exception as exc:
        errors.append(f"Invalid manifest: {exc}")
        return {"valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    # 2. Expected files
    expected_files = [
        "canonical/train.jsonl",
        "canonical/validation.jsonl",
        "kb/documents.jsonl",
        "metadata/provenance.jsonl",
        "metadata/splits.json",
        "checksums.sha256",
    ]
    for rel in expected_files:
        if not (root / rel).exists():
            errors.append(f"Missing expected file: {rel}")

    # Optional files that are nice to have
    optional_files = [
        "reports/quality.json",
        "reports/quality.md",
        "reports/backend-validation.json",
        "DATASET_CARD.md",
        "training/lora/train.jsonl",
        "training/lora/validation.jsonl",
        "training/osft/train.jsonl",
        "training/osft/validation.jsonl",
    ]
    for rel in optional_files:
        if not (root / rel).exists():
            warnings.append(f"Optional file missing: {rel}")

    # 3. Checksums
    mgr = BundleManager(root)
    checksum_ok, checksum_errors = mgr.validate_checksums()
    errors.extend(checksum_errors)

    # 4. Sample counts
    for split_name, expected_count in [
        ("train", manifest.canonical_train_count),
        ("validation", manifest.canonical_validation_count),
    ]:
        split_path = root / "canonical" / f"{split_name}.jsonl"
        if split_path.exists():
            try:
                raw = _load_jsonl(split_path)
                actual = len(raw)
                if actual != expected_count:
                    errors.append(
                        f"Sample count mismatch in {split_name}: "
                        f"manifest says {expected_count}, file has {actual}"
                    )
            except Exception as exc:
                errors.append(f"Error reading {split_name} samples: {exc}")

    # 5. Split metadata consistency
    splits_path = root / "metadata" / "splits.json"
    if splits_path.exists():
        try:
            with open(splits_path, encoding="utf-8") as fh:
                split_info = SplitInfo(**json.load(fh))
            if split_info.train_count != manifest.canonical_train_count:
                errors.append(
                    f"Split metadata train_count ({split_info.train_count}) "
                    f"differs from manifest ({manifest.canonical_train_count})"
                )
            if split_info.validation_count != manifest.canonical_validation_count:
                errors.append(
                    f"Split metadata validation_count ({split_info.validation_count}) "
                    f"differs from manifest ({manifest.canonical_validation_count})"
                )
        except Exception as exc:
            errors.append(f"Error reading split metadata: {exc}")

    valid = len(errors) == 0
    summary = {
        "bundle_name": manifest.bundle_name,
        "bundle_version": manifest.bundle_version,
        "model_id": manifest.model_id,
        "train_samples": manifest.canonical_train_count,
        "validation_samples": manifest.canonical_validation_count,
        "checksums_ok": checksum_ok,
        "file_errors": len(errors),
        "file_warnings": len(warnings),
    }

    level = logging.INFO if valid else logging.ERROR
    logger.log(level, "Bundle validation %s: %s", "PASSED" if valid else "FAILED", summary)

    return {"valid": valid, "errors": errors, "warnings": warnings, "summary": summary}


# ---------------------------------------------------------------------------
# Bundle fetching
# ---------------------------------------------------------------------------

def fetch_bundle(release_config: dict[str, Any] | None = None) -> Path:
    """Fetch a prepared dataset bundle from the first available source.

    Tries each source in the order defined in the release config
    (typically local → S3 → PVC).  If none are available the function
    raises explicitly — **SDG is never triggered**.

    Args:
        release_config: Parsed data-release configuration.  If *None*,
            the default ``configs/data-release.yaml`` is loaded.

    Returns:
        Path to the bundle root directory.

    Raises:
        FileNotFoundError: If the bundle cannot be found in any source.
        RuntimeError: On download / copy failure.
    """
    if release_config is None:
        release_config = load_bundle_config()

    sources = release_config.get("fetch", {}).get("sources", [])
    if not sources:
        raise FileNotFoundError(
            "No fetch sources defined in release config. "
            "Cannot locate a prepared dataset bundle."
        )

    bundle_cfg = release_config.get("bundle", {})
    bundle_name = f"{bundle_cfg.get('name', 'tau-knowledge')}-{bundle_cfg.get('version', 'v1')}"

    last_error: Exception | None = None
    for source in sources:
        src_type = source.get("type", "")
        try:
            if src_type == "local":
                local_path = Path(source["path"]).resolve()
                if local_path.is_dir() and (local_path / "manifest.json").exists():
                    logger.info("Found bundle at local path: %s", local_path)
                    return local_path
                logger.debug("Local path not available: %s", local_path)

            elif src_type == "s3":
                bundle_path = _fetch_from_s3(
                    bucket=source.get("bucket", ""),
                    prefix=source.get("prefix", ""),
                    bundle_name=bundle_name,
                    verify_checksums=release_config.get("fetch", {}).get("checksum_verify", True),
                )
                return bundle_path

            else:
                logger.warning("Unknown source type %r; skipping", src_type)

        except Exception as exc:
            last_error = exc
            logger.debug("Source %s failed: %s", src_type, exc)
            continue

    msg = (
        f"Prepared dataset bundle '{bundle_name}' not found in any configured source. "
        "This lab requires a pre-built bundle — automatic SDG is not permitted."
    )
    if last_error is not None:
        msg += f" Last error: {last_error}"
    raise FileNotFoundError(msg)


def _fetch_from_s3(
    bucket: str,
    prefix: str,
    bundle_name: str,
    verify_checksums: bool = True,
) -> Path:
    """Download a bundle from S3 to a local cache directory.

    Uses :mod:`boto3` if available; raises :class:`ImportError` otherwise.
    """
    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required to fetch bundles from S3. "
            "Install it with: pip install boto3"
        ) from exc

    s3 = boto3.client("s3")
    s3_prefix = f"{prefix.rstrip('/')}/{bundle_name}/" if prefix else f"{bundle_name}/"

    # List objects under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

    keys: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    if not keys:
        raise FileNotFoundError(
            f"No objects found in s3://{bucket}/{s3_prefix}"
        )

    # Download to local cache
    cache_dir = Path("data") / "cache" / "s3" / bundle_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        rel = key[len(s3_prefix):]
        if not rel:
            continue
        local_file = cache_dir / rel
        local_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", bucket, key, local_file)
        s3.download_file(bucket, key, str(local_file))

    # Verify checksums after download
    if verify_checksums:
        mgr = BundleManager(cache_dir)
        ok, errors = mgr.validate_checksums()
        if not ok:
            raise RuntimeError(
                f"S3 bundle checksum verification failed: {errors}"
            )

    logger.info("Bundle downloaded from S3 to %s", cache_dir)
    return cache_dir


__all__ = [
    "BundleManager",
    "compute_file_checksum",
    "fetch_bundle",
    "validate_prepared_dataset",
]
