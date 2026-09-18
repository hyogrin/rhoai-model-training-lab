"""Tests for MLflow tracking: LocalResultStore, MLflowTracker, fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhoai_model_training_lab.tracking import (
    LocalResultStore,
    MLflowTracker,
)
from rhoai_model_training_lab.schemas.evaluation import EvalResult, EvalMetrics
from rhoai_model_training_lab.schemas.training import TrainingResult


class TestLocalResultStore:
    """Test LocalResultStore save and load roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        record = {"type": "test", "value": 42, "nested": {"key": "value"}}
        fp = store.save("test_category", record)
        assert fp  # non-empty fingerprint

        loaded = store.load("test_category")
        assert len(loaded) == 1
        assert loaded[0]["type"] == "test"
        assert loaded[0]["value"] == 42
        assert loaded[0]["nested"]["key"] == "value"
        assert "__fingerprint" in loaded[0]
        assert "__saved_at" in loaded[0]

    def test_multiple_saves(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        store.save("multi", {"item": 1})
        store.save("multi", {"item": 2})
        store.save("multi", {"item": 3})
        loaded = store.load("multi")
        assert len(loaded) == 3
        items = [r["item"] for r in loaded]
        assert items == [1, 2, 3]

    def test_load_empty_category(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        loaded = store.load("nonexistent")
        assert loaded == []

    def test_save_eval_result(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        result = EvalResult(
            run_id="test-run-001",
            track="tau_episodes",
            variant="lora_rag",
            execution_status="completed",
        )
        fp = store.save_eval_result(result)
        assert fp
        loaded = store.load("evaluation")
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == "test-run-001"

    def test_save_training_result(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        result = TrainingResult(
            method="lora",
            model_id="test-model",
            final_train_loss=0.35,
        )
        fp = store.save_training_result(result)
        assert fp
        loaded = store.load("training")
        assert len(loaded) == 1
        assert loaded[0]["method"] == "lora"

    def test_load_pending(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        store.save("eval", {"record": "pending"})
        pending = store.load_pending("eval")
        assert len(pending) == 1
        assert "__mlflow_run_id" not in pending[0]

    def test_mark_uploaded(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        fp = store.save("eval", {"record": "to_upload"})
        store.mark_uploaded("eval", fp, "mlflow-run-abc")

        loaded = store.load("eval")
        assert len(loaded) == 1
        assert loaded[0]["__mlflow_run_id"] == "mlflow-run-abc"

        pending = store.load_pending("eval")
        assert len(pending) == 0

    def test_get_fingerprints(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        fp1 = store.save("dedup", {"data": "first"})
        fp2 = store.save("dedup", {"data": "second"})
        fps = store.get_fingerprints("dedup")
        assert fp1 in fps
        assert fp2 in fps


class TestFingerprintDeduplication:
    """Test fingerprint deduplication in LocalResultStore."""

    def test_same_data_same_fingerprint(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        fp1 = store.save("dedup", {"key": "value", "num": 42})
        fp2 = store.save("dedup", {"key": "value", "num": 42})
        assert fp1 == fp2

    def test_different_data_different_fingerprint(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        fp1 = store.save("dedup", {"key": "value_a"})
        fp2 = store.save("dedup", {"key": "value_b"})
        assert fp1 != fp2

    def test_can_detect_duplicates(self, tmp_path):
        store = LocalResultStore(base_dir=tmp_path)
        record = {"experiment": "test", "metric": 0.95}
        fp1 = store.save("check", record)
        known = store.get_fingerprints("check")
        fp2_candidate = store.save("check", record)
        assert fp2_candidate in known


class TestMLflowTracker:
    """Test MLflowTracker graceful failure when MLflow unavailable."""

    def test_no_mlflow_mode_saves_locally(self, tmp_path):
        local_store = LocalResultStore(base_dir=tmp_path)
        tracker = MLflowTracker(
            no_mlflow=True,
            local_store=local_store,
        )
        result = TrainingResult(method="lora", model_id="test")
        run_id = tracker.log_training_run(result)
        assert run_id == ""

        records = local_store.load("training")
        assert len(records) == 1

    def test_no_mlflow_eval_logging(self, tmp_path):
        local_store = LocalResultStore(base_dir=tmp_path)
        tracker = MLflowTracker(
            no_mlflow=True,
            local_store=local_store,
        )
        eval_result = EvalResult(
            run_id="eval-test",
            track="tau_episodes",
            variant="base",
            execution_status="completed",
        )
        run_id = tracker.log_eval_run(eval_result)
        assert run_id == ""

        records = local_store.load("evaluation")
        assert len(records) == 1

    def test_no_mlflow_data_release_logging(self, tmp_path):
        from rhoai_model_training_lab.schemas.data import BundleManifest, QualityReport

        local_store = LocalResultStore(base_dir=tmp_path)
        tracker = MLflowTracker(no_mlflow=True, local_store=local_store)
        manifest = BundleManifest(bundle_name="test", bundle_version="v1")
        quality = QualityReport(total_generated=100, total_accepted=90)
        run_id = tracker.log_data_release(manifest, quality)
        assert run_id == ""

        records = local_store.load("data")
        assert len(records) == 1

    def test_reupload_when_mlflow_disabled(self, tmp_path):
        local_store = LocalResultStore(base_dir=tmp_path)
        tracker = MLflowTracker(no_mlflow=True, local_store=local_store)
        result = tracker.reupload_pending("evaluation")
        assert result == []

    def test_tracker_ensures_local_persistence_first(self, tmp_path):
        """Even when MLflow is configured, local save happens first."""
        local_store = LocalResultStore(base_dir=tmp_path)
        tracker = MLflowTracker(
            tracking_uri="http://fake-mlflow:5000",
            no_mlflow=True,
            local_store=local_store,
        )
        result = TrainingResult(method="osft", model_id="test")
        tracker.log_training_run(result)
        assert len(local_store.load("training")) == 1
