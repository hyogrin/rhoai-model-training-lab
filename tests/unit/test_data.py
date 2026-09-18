"""Tests for the data module: checksums, bundle validation, BundleManager."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rhoai_model_training_lab.data import (
    BundleManager,
    compute_file_checksum,
    validate_prepared_dataset,
    fetch_bundle,
)


class TestComputeFileChecksum:
    """Test compute_file_checksum produces correct SHA256."""

    def test_known_content(self, tmp_path):
        content = b"hello, world!"
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_checksum(test_file) == expected

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_file_checksum(empty) == expected

    def test_binary_file(self, tmp_path):
        data = bytes(range(256))
        bin_file = tmp_path / "binary.bin"
        bin_file.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_checksum(bin_file) == expected

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_file_checksum(tmp_path / "missing.txt")

    def test_md5_algorithm(self, tmp_path):
        content = b"test content"
        f = tmp_path / "md5.txt"
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert compute_file_checksum(f, algorithm="md5") == expected


class TestValidatePreparedDataset:
    """Test validate_prepared_dataset catches common issues."""

    def test_valid_bundle(self, sample_bundle_path):
        result = validate_prepared_dataset(sample_bundle_path)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["summary"]["bundle_name"] == "test-bundle"

    def test_missing_manifest(self, tmp_path):
        bundle = tmp_path / "bad-bundle"
        bundle.mkdir()
        result = validate_prepared_dataset(bundle)
        assert result["valid"] is False
        assert any("manifest.json not found" in e for e in result["errors"])

    def test_checksum_mismatch(self, sample_bundle_path):
        train_file = sample_bundle_path / "canonical" / "train.jsonl"
        train_file.write_text('{"sample_id": "tampered", "messages": [], "sample_type": "policy_qa"}\n')
        result = validate_prepared_dataset(sample_bundle_path)
        assert result["valid"] is False
        assert any("Checksum mismatch" in e for e in result["errors"])

    def test_sample_count_mismatch(self, sample_bundle_path):
        manifest_path = sample_bundle_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["canonical_train_count"] = 99
        manifest_path.write_text(json.dumps(manifest))
        result = validate_prepared_dataset(sample_bundle_path)
        assert result["valid"] is False
        assert any("Sample count mismatch" in e for e in result["errors"])

    def test_missing_expected_files(self, sample_bundle_path):
        (sample_bundle_path / "kb" / "documents.jsonl").unlink()
        result = validate_prepared_dataset(sample_bundle_path)
        assert result["valid"] is False
        assert any("Missing expected file" in e for e in result["errors"])


class TestBundleManager:
    """Test BundleManager loads samples correctly."""

    def test_load_bundle(self, sample_bundle_path):
        mgr = BundleManager.load_bundle(sample_bundle_path)
        assert mgr.manifest.bundle_name == "test-bundle"
        assert mgr.manifest.canonical_train_count == 1

    def test_get_train_samples(self, sample_bundle_path):
        mgr = BundleManager(sample_bundle_path)
        samples = mgr.get_samples("train")
        assert len(samples) == 1
        assert samples[0].sample_id == "train-0001"

    def test_get_validation_samples(self, sample_bundle_path):
        mgr = BundleManager(sample_bundle_path)
        samples = mgr.get_samples("validation")
        assert len(samples) == 1
        assert samples[0].sample_id == "val-0001"

    def test_invalid_split_raises(self, sample_bundle_path):
        mgr = BundleManager(sample_bundle_path)
        with pytest.raises(ValueError, match="split must be one of"):
            mgr.get_samples("test")

    def test_validate_checksums_pass(self, sample_bundle_path):
        mgr = BundleManager(sample_bundle_path)
        ok, errors = mgr.validate_checksums()
        assert ok is True
        assert errors == []

    def test_validate_checksums_detect_corruption(self, sample_bundle_path):
        train_file = sample_bundle_path / "canonical" / "train.jsonl"
        train_file.write_text("corrupted content\n")
        mgr = BundleManager(sample_bundle_path)
        ok, errors = mgr.validate_checksums()
        assert ok is False
        assert len(errors) > 0

    def test_split_info(self, sample_bundle_path):
        mgr = BundleManager(sample_bundle_path)
        si = mgr.split_info
        assert si.method == "scenario_family"
        assert si.train_count == 1
        assert si.validation_count == 1

    def test_missing_manifest_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No manifest.json"):
            BundleManager.load_bundle(empty_dir)

    def test_get_training_samples(self, sample_bundle_path):
        mgr = BundleManager(sample_bundle_path)
        lora_train = mgr.get_training_samples("lora", "train")
        assert len(lora_train) == 1
        assert "messages" in lora_train[0]

        osft_val = mgr.get_training_samples("osft", "validation")
        assert len(osft_val) == 1


class TestFetchBundle:
    """Test fetch_bundle never triggers SDG (mock external calls)."""

    def test_fetch_from_local_source(self, sample_bundle_path):
        release_config = {
            "fetch": {
                "sources": [
                    {"type": "local", "path": str(sample_bundle_path)},
                ],
            },
            "bundle": {"name": "test-bundle", "version": "v1"},
        }
        result = fetch_bundle(release_config)
        assert result == sample_bundle_path.resolve()

    def test_no_sources_raises(self):
        release_config = {"fetch": {"sources": []}}
        with pytest.raises(FileNotFoundError, match="No fetch sources"):
            fetch_bundle(release_config)

    def test_bundle_not_found_raises_no_sdg(self, tmp_path):
        release_config = {
            "fetch": {
                "sources": [
                    {"type": "local", "path": str(tmp_path / "nonexistent")},
                ],
            },
            "bundle": {"name": "test", "version": "v1"},
        }
        with pytest.raises(FileNotFoundError, match="not found in any configured source"):
            fetch_bundle(release_config)

    def test_sdg_never_triggered(self, tmp_path):
        """Ensure that fetch_bundle never imports or calls SDG modules."""
        release_config = {
            "fetch": {
                "sources": [
                    {"type": "local", "path": str(tmp_path / "nonexistent")},
                ],
            },
            "bundle": {"name": "test", "version": "v1"},
        }
        with patch("rhoai_model_training_lab.sdg.SDGPipeline") as mock_sdg:
            with pytest.raises(FileNotFoundError):
                fetch_bundle(release_config)
            mock_sdg.assert_not_called()
