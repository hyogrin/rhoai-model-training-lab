"""MLflow integration and local result persistence.

Provides ``MLflowTracker`` for logging data releases, training runs, and
evaluation results to MLflow, with graceful offline fallback.
``LocalResultStore`` handles JSONL persistence with fingerprint-based
deduplication and supports later re-upload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rhoai_model_training_lab.config import PROJECT_ROOT
from rhoai_model_training_lab.schemas.data import BundleManifest, QualityReport
from rhoai_model_training_lab.schemas.evaluation import (
    ComparisonRow,
    EvalResult,
)
from rhoai_model_training_lab.schemas.training import TrainingResult

logger = logging.getLogger(__name__)

__all__ = [
    "MLflowTracker",
    "LocalResultStore",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_DATA = "rhoai-model-training-lab-data"
EXPERIMENT_TRAINING = "rhoai-model-training-lab-training"
EXPERIMENT_EVALUATION = "rhoai-model-training-lab-evaluation"

_DEFAULT_LOCAL_DIR = "data/tracking/local_results"
_ARTIFACT_SIZE_LIMIT_MB = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_fingerprint(data: dict[str, Any]) -> str:
    """SHA-256 fingerprint of a dict for deduplication."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _safe_metric_value(value: Any) -> float | None:
    """Convert a value to a float suitable for MLflow, or None."""
    if value is None:
        return None
    try:
        fval = float(value)
        if fval != fval:  # NaN check
            return None
        return fval
    except (TypeError, ValueError):
        return None


def _is_mlflow_available() -> bool:
    """Check if the mlflow package is importable."""
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Local Result Store
# ---------------------------------------------------------------------------

class LocalResultStore:
    """Persistent JSONL storage for evaluation and training results.

    Saves results locally with fingerprints so they can be de-duplicated
    and re-uploaded to MLflow later.

    Args:
        base_dir: Directory for local JSONL files.  Defaults to
            ``data/tracking/local_results`` under the project root.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            self.base_dir = PROJECT_ROOT / _DEFAULT_LOCAL_DIR
        else:
            self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _store_path(self, category: str) -> Path:
        return self.base_dir / f"{category}.jsonl"

    # -- write --------------------------------------------------------------

    def save(self, category: str, record: dict[str, Any]) -> str:
        """Append a record to the category JSONL file.

        A ``__fingerprint`` and ``__saved_at`` field are added automatically.

        Args:
            category: Logical category (e.g. 'data', 'training', 'evaluation').
            record: The record dict to persist.

        Returns:
            The computed fingerprint string.
        """
        fp = _compute_fingerprint(record)
        record = {**record, "__fingerprint": fp, "__saved_at": datetime.now(timezone.utc).isoformat()}
        path = self._store_path(category)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.debug("Saved record (fp=%s) to %s", fp, path)
        return fp

    def save_eval_result(self, result: EvalResult) -> str:
        """Convenience: save an EvalResult."""
        return self.save("evaluation", result.model_dump(mode="json"))

    def save_training_result(self, result: TrainingResult) -> str:
        """Convenience: save a TrainingResult."""
        return self.save("training", result.model_dump(mode="json"))

    # -- read ---------------------------------------------------------------

    def load(self, category: str) -> list[dict[str, Any]]:
        """Load all records for a category.

        Args:
            category: Logical category name.

        Returns:
            List of record dicts (including metadata fields).
        """
        path = self._store_path(category)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def load_pending(self, category: str) -> list[dict[str, Any]]:
        """Load records that have not been uploaded to MLflow.

        A record is considered pending if it lacks an ``__mlflow_run_id`` field.

        Args:
            category: Logical category name.

        Returns:
            List of pending record dicts.
        """
        all_records = self.load(category)
        return [r for r in all_records if not r.get("__mlflow_run_id")]

    def mark_uploaded(self, category: str, fingerprint: str, mlflow_run_id: str) -> None:
        """Mark a record as uploaded by rewriting the JSONL with the run ID.

        Args:
            category: Logical category name.
            fingerprint: Fingerprint of the record to update.
            mlflow_run_id: The MLflow run ID that was created.
        """
        path = self._store_path(category)
        if not path.exists():
            return

        updated_lines: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    updated_lines.append(line)
                    continue
                if record.get("__fingerprint") == fingerprint:
                    record["__mlflow_run_id"] = mlflow_run_id
                    record["__uploaded_at"] = datetime.now(timezone.utc).isoformat()
                updated_lines.append(json.dumps(record, default=str))

        with open(path, "w") as f:
            for uline in updated_lines:
                f.write(uline + "\n")

    def get_fingerprints(self, category: str) -> set[str]:
        """Get all known fingerprints for deduplication.

        Args:
            category: Logical category name.

        Returns:
            Set of fingerprint strings.
        """
        records = self.load(category)
        return {r["__fingerprint"] for r in records if "__fingerprint" in r}


# ---------------------------------------------------------------------------
# MLflow Tracker
# ---------------------------------------------------------------------------

class MLflowTracker:
    """MLflow integration for the model training lab.

    Logs data releases, training runs, and evaluation results to MLflow
    with proper experiment naming, lineage linking, and artifact handling.

    Supports ``--no-mlflow`` offline mode: always saves local results first,
    then attempts MLflow logging.  Failures are recorded but never prevent
    local persistence.

    Args:
        tracking_uri: MLflow tracking server URI.  Defaults to
            ``$MLFLOW_TRACKING_URI`` or ``./mlruns``.
        no_mlflow: If True, skip all MLflow calls (offline mode).
        local_store: Optional pre-configured LocalResultStore instance.
        artifact_max_size_mb: Max artifact size in MB before skipping upload.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        no_mlflow: bool = False,
        local_store: LocalResultStore | None = None,
        artifact_max_size_mb: int = _ARTIFACT_SIZE_LIMIT_MB,
    ) -> None:
        self.no_mlflow = no_mlflow
        self.local_store = local_store or LocalResultStore()
        self.artifact_max_size_mb = artifact_max_size_mb
        self._tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        self._mlflow_initialized = False

    def _ensure_mlflow(self) -> Any:
        """Import and configure mlflow, returning the module.

        Raises ImportError if mlflow is not installed.
        """
        if self.no_mlflow:
            raise RuntimeError("MLflow is disabled (--no-mlflow mode)")

        import mlflow

        if not self._mlflow_initialized:
            mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow_initialized = True
            logger.info("MLflow tracking URI: %s", self._tracking_uri)

        return mlflow

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create an MLflow experiment, returning its ID."""
        mlflow = self._ensure_mlflow()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return mlflow.create_experiment(experiment_name)

    # -- Data Release -------------------------------------------------------

    def log_data_release(
        self,
        manifest: BundleManifest,
        quality_report: QualityReport,
    ) -> str:
        """Log a data release to the data experiment.

        Args:
            manifest: The bundle manifest.
            quality_report: Quality assessment of the generated data.

        Returns:
            MLflow run ID, or empty string if logging failed/skipped.
        """
        record = {
            "type": "data_release",
            "manifest": manifest.model_dump(mode="json"),
            "quality_report": quality_report.model_dump(mode="json"),
        }
        fp = self.local_store.save("data", record)

        if self.no_mlflow:
            logger.info("MLflow disabled; data release saved locally (fp=%s)", fp)
            return ""

        try:
            mlflow = self._ensure_mlflow()
            experiment_id = self._get_or_create_experiment(EXPERIMENT_DATA)

            with mlflow.start_run(experiment_id=experiment_id, run_name=f"data-{manifest.bundle_name}") as run:
                mlflow.log_params({
                    "bundle_name": manifest.bundle_name,
                    "bundle_version": manifest.bundle_version,
                    "tau_version": manifest.tau_version,
                    "tau_commit_sha": manifest.tau_commit_sha[:12] if manifest.tau_commit_sha else "",
                    "model_id": manifest.model_id,
                    "model_revision": manifest.model_revision,
                    "split_policy": manifest.split_policy,
                    "bundle_hash": manifest.bundle_hash[:16] if manifest.bundle_hash else "",
                })

                metrics: dict[str, float] = {}
                m = _safe_metric_value(manifest.canonical_train_count)
                if m is not None:
                    metrics["canonical_train_count"] = m
                m = _safe_metric_value(manifest.canonical_validation_count)
                if m is not None:
                    metrics["canonical_validation_count"] = m
                m = _safe_metric_value(quality_report.total_generated)
                if m is not None:
                    metrics["total_generated"] = m
                m = _safe_metric_value(quality_report.total_accepted)
                if m is not None:
                    metrics["total_accepted"] = m
                m = _safe_metric_value(quality_report.total_rejected)
                if m is not None:
                    metrics["total_rejected"] = m
                m = _safe_metric_value(quality_report.acceptance_rate)
                if m is not None:
                    metrics["acceptance_rate"] = m

                if metrics:
                    mlflow.log_metrics(metrics)

                self._log_json_artifact(mlflow, manifest.model_dump(mode="json"), "manifest.json")
                self._log_json_artifact(mlflow, quality_report.model_dump(mode="json"), "quality_report.json")

                run_id = run.info.run_id
                self.local_store.mark_uploaded("data", fp, run_id)
                logger.info("Data release logged to MLflow run %s", run_id)
                return run_id

        except Exception as exc:
            logger.error("MLflow data release logging failed: %s", exc)
            logger.info("Data release saved locally (fp=%s); re-upload later with log-results", fp)
            return ""

    # -- Training Run -------------------------------------------------------

    def log_training_run(
        self,
        training_result: TrainingResult,
        *,
        data_run_id: str = "",
    ) -> str:
        """Log a training run to the training experiment.

        Links back to the data run via ``data_run_id``.  Never auto-uploads
        multi-GB model weights; instead records the PVC/S3/OCI URI and hash.

        Args:
            training_result: The training result to log.
            data_run_id: MLflow run ID of the data release (for lineage).

        Returns:
            MLflow run ID, or empty string if logging failed/skipped.
        """
        record = {
            "type": "training_run",
            "result": training_result.model_dump(mode="json"),
            "data_run_id": data_run_id,
        }
        fp = self.local_store.save("training", record)

        if self.no_mlflow:
            logger.info("MLflow disabled; training result saved locally (fp=%s)", fp)
            return ""

        try:
            mlflow = self._ensure_mlflow()
            experiment_id = self._get_or_create_experiment(EXPERIMENT_TRAINING)

            run_name = f"train-{training_result.method}-{training_result.bundle_id[:8]}"
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                mlflow.log_params({
                    "method": training_result.method,
                    "model_id": training_result.model_id,
                    "model_revision": training_result.model_revision,
                    "bundle_id": training_result.bundle_id,
                    "bundle_hash": training_result.bundle_hash[:16] if training_result.bundle_hash else "",
                    "canonical_hash": training_result.canonical_hash[:16] if training_result.canonical_hash else "",
                    "export_hash": training_result.export_hash[:16] if training_result.export_hash else "",
                    "seed": str(training_result.seed),
                    "gpu_name": training_result.gpu_name,
                })

                if data_run_id:
                    mlflow.set_tag("data_run_id", data_run_id)
                mlflow.set_tag("mlflow.note.content", f"Training: {training_result.method}")

                metrics: dict[str, float] = {}
                for field_name in (
                    "train_samples", "validation_samples", "total_steps",
                    "total_tokens", "num_epochs", "final_train_loss",
                    "final_eval_loss", "best_eval_loss", "best_checkpoint_step",
                    "wall_time_seconds", "peak_vram_gb",
                ):
                    val = _safe_metric_value(getattr(training_result, field_name, None))
                    if val is not None:
                        metrics[field_name] = val

                if metrics:
                    mlflow.log_metrics(metrics)

                weight_paths = {
                    "checkpoint_path": training_result.checkpoint_path,
                    "adapter_path": training_result.adapter_path,
                    "merged_path": training_result.merged_path,
                    "exported_path": training_result.exported_path,
                }
                for key, path_val in weight_paths.items():
                    if path_val:
                        mlflow.set_tag(f"weight_{key}", path_val)
                        logger.info(
                            "Recorded weight URI for %s: %s (not auto-uploaded)",
                            key,
                            path_val,
                        )

                self._log_json_artifact(
                    mlflow,
                    training_result.model_dump(mode="json"),
                    "training_result.json",
                )

                run_id = run.info.run_id
                self.local_store.mark_uploaded("training", fp, run_id)
                logger.info("Training run logged to MLflow run %s", run_id)
                return run_id

        except Exception as exc:
            logger.error("MLflow training logging failed: %s", exc)
            logger.info("Training result saved locally (fp=%s); re-upload later", fp)
            return ""

    # -- Evaluation Run -----------------------------------------------------

    def log_eval_run(
        self,
        eval_result: EvalResult,
        *,
        training_run_id: str = "",
    ) -> str:
        """Log an evaluation run to the evaluation experiment.

        Links back to the training run via ``training_run_id``.

        Args:
            eval_result: The evaluation result to log.
            training_run_id: MLflow run ID of the training run (for lineage).

        Returns:
            MLflow run ID, or empty string if logging failed/skipped.
        """
        record = {
            "type": "eval_run",
            "result": eval_result.model_dump(mode="json"),
            "training_run_id": training_run_id,
        }
        fp = self.local_store.save("evaluation", record)

        if self.no_mlflow:
            logger.info("MLflow disabled; eval result saved locally (fp=%s)", fp)
            return ""

        try:
            mlflow = self._ensure_mlflow()
            experiment_id = self._get_or_create_experiment(EXPERIMENT_EVALUATION)

            run_name = f"eval-{eval_result.track}-{eval_result.variant}"
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                params: dict[str, str] = {
                    "track": eval_result.track,
                    "variant": eval_result.variant,
                    "execution_status": eval_result.execution_status,
                    "verification_status": eval_result.verification_status,
                    "model_hash": eval_result.model_hash[:16] if eval_result.model_hash else "",
                    "bundle_id": eval_result.bundle_id,
                    "grader_version": eval_result.grader_version,
                    "scorer_revision": eval_result.scorer_revision,
                }
                if eval_result.config:
                    params.update({
                        "knowledge_access": eval_result.config.knowledge_access,
                        "mode": eval_result.config.mode,
                        "trials": str(eval_result.config.trials),
                        "seed": str(eval_result.config.seed),
                    })
                mlflow.log_params(params)

                if training_run_id:
                    mlflow.set_tag("training_run_id", training_run_id)
                if eval_result.data_run_id:
                    mlflow.set_tag("data_run_id", eval_result.data_run_id)

                if eval_result.metrics:
                    m = eval_result.metrics
                    metrics: dict[str, float] = {}
                    for field_name in (
                        "total_tasks", "attempted", "completed", "succeeded",
                        "failed", "unsupported", "timed_out", "budget_exceeded",
                        "task_success_rate", "pass_k", "action_recall",
                        "answer_accuracy", "tool_name_accuracy",
                        "tool_args_accuracy", "grounding_score",
                        "condition_accuracy", "ci_lower", "ci_upper",
                        "mean_turns", "mean_tool_calls", "mean_tokens",
                        "mean_wall_time_seconds", "mean_retrieval_latency_ms",
                        "mean_model_latency_ms",
                    ):
                        val = _safe_metric_value(getattr(m, field_name, None))
                        if val is not None:
                            metrics[field_name] = val
                    if metrics:
                        mlflow.log_metrics(metrics)

                for rr in eval_result.retention_results:
                    mlflow.log_metrics({
                        f"retention_{rr.variant}_accuracy": rr.accuracy,
                        f"retention_{rr.variant}_delta_pp": rr.retention_delta_pp,
                    })

                self._log_json_artifact(
                    mlflow,
                    eval_result.model_dump(mode="json"),
                    "eval_result.json",
                )

                if eval_result.task_results:
                    task_data = [tr.model_dump(mode="json") for tr in eval_result.task_results]
                    self._log_jsonl_artifact(mlflow, task_data, "task_results.jsonl")

                if eval_result.retention_results:
                    ret_data = [rr.model_dump(mode="json") for rr in eval_result.retention_results]
                    self._log_jsonl_artifact(mlflow, ret_data, "retention_results.jsonl")

                if eval_result.protocol_deviations:
                    self._log_json_artifact(
                        mlflow,
                        {"deviations": eval_result.protocol_deviations},
                        "protocol_deviations.json",
                    )

                run_id = run.info.run_id
                self.local_store.mark_uploaded("evaluation", fp, run_id)
                logger.info("Eval run logged to MLflow run %s", run_id)
                return run_id

        except Exception as exc:
            logger.error("MLflow eval logging failed: %s", exc)
            logger.info("Eval result saved locally (fp=%s); re-upload later", fp)
            return ""

    # -- Comparison Table ---------------------------------------------------

    def log_comparison(
        self,
        comparison_rows: list[ComparisonRow],
        *,
        parent_run_ids: list[str] | None = None,
    ) -> str:
        """Log a comparison table as an MLflow artifact.

        Args:
            comparison_rows: Rows from the comparison table.
            parent_run_ids: Optional list of related eval run IDs.

        Returns:
            MLflow run ID, or empty string if logging failed/skipped.
        """
        table_data = [row.model_dump(mode="json") for row in comparison_rows]
        record = {
            "type": "comparison",
            "rows": table_data,
            "parent_run_ids": parent_run_ids or [],
        }
        fp = self.local_store.save("comparison", record)

        if self.no_mlflow:
            logger.info("MLflow disabled; comparison table saved locally (fp=%s)", fp)
            return ""

        try:
            mlflow = self._ensure_mlflow()
            experiment_id = self._get_or_create_experiment(EXPERIMENT_EVALUATION)

            with mlflow.start_run(experiment_id=experiment_id, run_name="comparison") as run:
                mlflow.set_tag("result_type", "comparison_table")
                if parent_run_ids:
                    mlflow.set_tag("parent_run_ids", json.dumps(parent_run_ids))

                self._log_json_artifact(mlflow, table_data, "comparison_table.json")

                try:
                    import pandas as pd

                    df = pd.DataFrame(table_data)
                    csv_path = self._temp_artifact_path("comparison_table.csv")
                    df.to_csv(csv_path, index=False)
                    mlflow.log_artifact(str(csv_path), "tables")
                except ImportError:
                    logger.debug("pandas not available; skipping CSV artifact")

                run_id = run.info.run_id
                self.local_store.mark_uploaded("comparison", fp, run_id)
                logger.info("Comparison table logged to MLflow run %s", run_id)
                return run_id

        except Exception as exc:
            logger.error("MLflow comparison logging failed: %s", exc)
            return ""

    # -- Re-upload (offline → MLflow) ---------------------------------------

    def reupload_pending(self, category: str = "evaluation") -> list[str]:
        """Re-upload locally saved results that were not yet in MLflow.

        Used by the ``log-results`` CLI command to recover from offline runs.

        Args:
            category: Which category to re-upload.

        Returns:
            List of newly created MLflow run IDs.
        """
        if self.no_mlflow:
            logger.warning("Cannot re-upload: MLflow is disabled")
            return []

        pending = self.local_store.load_pending(category)
        if not pending:
            logger.info("No pending records in category '%s'", category)
            return []

        logger.info("Found %d pending records in '%s'", len(pending), category)
        uploaded_ids: list[str] = []

        for record in pending:
            fp = record.get("__fingerprint", "")
            record_type = record.get("type", "")

            try:
                if record_type == "eval_run" and "result" in record:
                    eval_result = EvalResult(**record["result"])
                    run_id = self.log_eval_run(
                        eval_result,
                        training_run_id=record.get("training_run_id", ""),
                    )
                    if run_id:
                        uploaded_ids.append(run_id)

                elif record_type == "training_run" and "result" in record:
                    training_result = TrainingResult(**record["result"])
                    run_id = self.log_training_run(
                        training_result,
                        data_run_id=record.get("data_run_id", ""),
                    )
                    if run_id:
                        uploaded_ids.append(run_id)

                elif record_type == "data_release":
                    manifest = BundleManifest(**record["manifest"])
                    quality = QualityReport(**record["quality_report"])
                    run_id = self.log_data_release(manifest, quality)
                    if run_id:
                        uploaded_ids.append(run_id)

                elif record_type == "comparison":
                    rows = [ComparisonRow(**r) for r in record.get("rows", [])]
                    run_id = self.log_comparison(
                        rows,
                        parent_run_ids=record.get("parent_run_ids", []),
                    )
                    if run_id:
                        uploaded_ids.append(run_id)

                else:
                    logger.warning("Unknown record type '%s' (fp=%s); skipping", record_type, fp)

            except Exception as exc:
                logger.error("Failed to re-upload record fp=%s: %s", fp, exc)

        logger.info("Re-uploaded %d/%d pending records", len(uploaded_ids), len(pending))
        return uploaded_ids

    # -- Artifact helpers ---------------------------------------------------

    @staticmethod
    def _temp_artifact_path(filename: str) -> Path:
        """Create a temporary artifact path under the project's tracking dir."""
        tmp_dir = PROJECT_ROOT / "data" / "tracking" / ".tmp_artifacts"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir / filename

    def _log_json_artifact(self, mlflow: Any, data: Any, filename: str) -> None:
        """Write JSON data to a temp file and log as an MLflow artifact."""
        path = self._temp_artifact_path(filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.artifact_max_size_mb:
            logger.warning(
                "Artifact %s is %.1f MB (limit %d MB); skipping upload",
                filename,
                size_mb,
                self.artifact_max_size_mb,
            )
            return

        mlflow.log_artifact(str(path), "results")

    def _log_jsonl_artifact(self, mlflow: Any, records: Sequence[dict[str, Any]], filename: str) -> None:
        """Write JSONL data to a temp file and log as an MLflow artifact."""
        path = self._temp_artifact_path(filename)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.artifact_max_size_mb:
            logger.warning(
                "Artifact %s is %.1f MB (limit %d MB); skipping upload",
                filename,
                size_mb,
                self.artifact_max_size_mb,
            )
            return

        mlflow.log_artifact(str(path), "results")
