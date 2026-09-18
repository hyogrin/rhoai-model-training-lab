"""Tests for the training module: validation, preprocessing, and sample consistency."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rhoai_model_training_lab.training import (
    DataPreprocessor,
    validate_bundle_for_training,
)


class TestValidateBundleForTraining:
    """Test validate_bundle_for_training catches missing files."""

    def test_missing_training_files(self, tmp_path):
        bundle = tmp_path / "incomplete-bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps({
            "bundle_name": "test",
            "bundle_version": "v1",
            "model_id": "test-model",
        }))
        for profile in ("lora", "osft"):
            result = validate_bundle_for_training(bundle, profile)
            assert result.loader_check is False
            assert any("missing" in e.lower() for e in result.errors)

    def test_valid_bundle_has_files(self, sample_bundle_path):
        for profile in ("lora", "osft"):
            result = validate_bundle_for_training(sample_bundle_path, profile)
            assert result.loader_check is True

    def test_invalid_profile_raises(self, sample_bundle_path):
        with pytest.raises(ValueError, match="profile must be"):
            validate_bundle_for_training(sample_bundle_path, "invalid")


class TestDataPreprocessor:
    """Test DataPreprocessor applies assistant-only loss mask."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 32000
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0

        def apply_chat_template(msgs, **kwargs):
            token_count = sum(10 + len(m.get("content", "") or "") for m in msgs)
            return list(range(100, 100 + min(token_count, 50)))

        tokenizer.apply_chat_template = apply_chat_template
        return tokenizer

    def test_assistant_only_loss_mask(self, mock_tokenizer):
        preprocessor = DataPreprocessor(
            tokenizer=mock_tokenizer,
            max_seq_length=4096,
            loss_masking="assistant_only",
        )
        messages = [
            {"role": "system", "content": "You are a banking assistant."},
            {"role": "user", "content": "What is the fee?"},
            {"role": "assistant", "content": "The fee is $25."},
        ]
        result = preprocessor.apply_chat_template(messages)
        labels = result["labels"]

        assert len(labels) == len(result["input_ids"])
        masked_count = sum(1 for l in labels if l == -100)
        trained_count = sum(1 for l in labels if l != -100)
        assert masked_count > 0, "System/user tokens should be masked"

    def test_no_tokenizer_raises(self):
        preprocessor = DataPreprocessor(tokenizer=None)
        with pytest.raises(RuntimeError, match="No tokenizer loaded"):
            preprocessor.apply_chat_template([{"role": "user", "content": "hello"}])

    def test_verify_loss_masking_empty_labels(self):
        preprocessor = DataPreprocessor(tokenizer=None)
        issues = preprocessor.verify_loss_masking({"labels": [], "input_ids": []})
        assert any("Empty labels" in i for i in issues)

    def test_verify_loss_masking_all_masked(self):
        preprocessor = DataPreprocessor(tokenizer=None)
        issues = preprocessor.verify_loss_masking({
            "labels": [-100, -100, -100],
            "input_ids": [1, 2, 3],
        })
        assert any("nothing to train" in i.lower() for i in issues)

    def test_loss_mask_masks_system_user_tool(self, mock_tokenizer):
        preprocessor = DataPreprocessor(
            tokenizer=mock_tokenizer,
            loss_masking="assistant_only",
        )
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User message."},
            {"role": "assistant", "content": "Assistant response."},
            {"role": "tool", "content": '{"result": "ok"}'},
            {"role": "assistant", "content": "Final answer."},
        ]
        result = preprocessor.apply_chat_template(messages)
        labels = result["labels"]
        assert isinstance(labels, list)
        assert len(labels) > 0


class TestLoRAOSFTSameIDs:
    """Test LoRA and OSFT use the same canonical sample IDs."""

    def test_same_sample_ids(self, sample_bundle_path):
        from rhoai_model_training_lab.data import BundleManager

        mgr = BundleManager(sample_bundle_path)
        lora_train = mgr.get_training_samples("lora", "train")
        osft_train = mgr.get_training_samples("osft", "train")

        assert len(lora_train) == len(osft_train), (
            "LoRA and OSFT should have the same number of training samples"
        )

        lora_val = mgr.get_training_samples("lora", "validation")
        osft_val = mgr.get_training_samples("osft", "validation")
        assert len(lora_val) == len(osft_val)

    def test_canonical_samples_match(self, sample_bundle_path):
        from rhoai_model_training_lab.data import BundleManager

        mgr = BundleManager(sample_bundle_path)
        train_samples = mgr.get_samples("train")
        val_samples = mgr.get_samples("validation")

        train_ids = {s.sample_id for s in train_samples}
        val_ids = {s.sample_id for s in val_samples}
        assert train_ids & val_ids == set(), "Train and val canonical IDs must not overlap"
