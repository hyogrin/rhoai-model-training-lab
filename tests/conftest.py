"""Shared pytest fixtures for the rhoai-model-training-lab test suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rhoai_model_training_lab.schemas.data import (
    CanonicalSample,
    Message,
    SampleType,
    ToolCall,
    ToolCallFunction,
    ToolSchema,
    ToolFunctionDef,
    ValidationStatus,
)


@pytest.fixture
def sample_message() -> Message:
    """Return a simple assistant Message."""
    return Message(role="assistant", content="Your account balance is $1,234.56.")


@pytest.fixture
def sample_tool_call_message() -> Message:
    """Return an assistant Message with tool_calls."""
    return Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_001",
                type="function",
                function=ToolCallFunction(
                    name="get_account_balance",
                    arguments='{"account_id": "ACC-12345"}',
                ),
            )
        ],
    )


@pytest.fixture
def sample_canonical() -> CanonicalSample:
    """Return a minimal CanonicalSample for testing."""
    return CanonicalSample(
        sample_id="test-policy-0001",
        messages=[
            Message(role="system", content="You are a banking assistant."),
            Message(role="user", content="What is the interest rate for savings?"),
            Message(role="assistant", content="The current savings interest rate is 4.5% APY."),
        ],
        source_doc_ids=["doc_savings_001"],
        scenario_family="savings_interest",
        sample_type=SampleType.POLICY_QA,
        validation_status=ValidationStatus.ACCEPTED,
    )


@pytest.fixture
def sample_bundle_path(tmp_path: Path) -> Path:
    """Create a minimal bundle directory structure and return its path.

    Layout:
        bundle_root/
        ├── manifest.json
        ├── checksums.sha256
        ├── canonical/
        │   ├── train.jsonl
        │   └── validation.jsonl
        ├── kb/
        │   └── documents.jsonl
        ├── metadata/
        │   ├── provenance.jsonl
        │   └── splits.json
        └── training/
            ├── lora/
            │   ├── train.jsonl
            │   └── validation.jsonl
            └── osft/
                ├── train.jsonl
                └── validation.jsonl
    """
    bundle = tmp_path / "test-bundle-v1"
    bundle.mkdir()

    train_sample = {
        "sample_id": "train-0001",
        "messages": [
            {"role": "system", "content": "You are a banking assistant."},
            {"role": "user", "content": "What is the late fee?"},
            {"role": "assistant", "content": "The late fee is $25."},
        ],
        "sample_type": "policy_qa",
        "scenario_family": "fees",
    }
    val_sample = {
        "sample_id": "val-0001",
        "messages": [
            {"role": "system", "content": "You are a banking assistant."},
            {"role": "user", "content": "How do I open an account?"},
            {"role": "assistant", "content": "Visit any branch with valid ID."},
        ],
        "sample_type": "policy_qa",
        "scenario_family": "account_opening",
    }
    kb_doc = {
        "doc_id": "kb_001",
        "text": "Late fees are charged at $25 per occurrence.",
    }

    # canonical/
    canonical_dir = bundle / "canonical"
    canonical_dir.mkdir()
    (canonical_dir / "train.jsonl").write_text(json.dumps(train_sample) + "\n")
    (canonical_dir / "validation.jsonl").write_text(json.dumps(val_sample) + "\n")

    # kb/
    kb_dir = bundle / "kb"
    kb_dir.mkdir()
    (kb_dir / "documents.jsonl").write_text(json.dumps(kb_doc) + "\n")

    # metadata/
    meta_dir = bundle / "metadata"
    meta_dir.mkdir()
    (meta_dir / "provenance.jsonl").write_text(
        json.dumps({"sample_id": "train-0001", "source_documents": ["kb_001"]}) + "\n"
    )
    splits = {
        "method": "scenario_family",
        "seed": 42,
        "train_ids": ["train-0001"],
        "validation_ids": ["val-0001"],
        "train_count": 1,
        "validation_count": 1,
        "family_partition": {"fees": "train", "account_opening": "validation"},
    }
    (meta_dir / "splits.json").write_text(json.dumps(splits))

    # training/
    for profile in ("lora", "osft"):
        profile_dir = bundle / "training" / profile
        profile_dir.mkdir(parents=True)
        (profile_dir / "train.jsonl").write_text(
            json.dumps({"messages": train_sample["messages"]}) + "\n"
        )
        (profile_dir / "validation.jsonl").write_text(
            json.dumps({"messages": val_sample["messages"]}) + "\n"
        )

    # checksums.sha256
    checksums_lines: list[str] = []
    for file_path in sorted(bundle.rglob("*")):
        if file_path.is_file() and file_path.name not in ("checksums.sha256", "manifest.json"):
            rel = str(file_path.relative_to(bundle))
            h = hashlib.sha256(file_path.read_bytes()).hexdigest()
            checksums_lines.append(f"{h} {rel}")
    (bundle / "checksums.sha256").write_text("\n".join(checksums_lines) + "\n")

    # manifest.json
    manifest = {
        "bundle_name": "test-bundle",
        "bundle_version": "v1",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "canonical_train_count": 1,
        "canonical_validation_count": 1,
        "split_policy": "scenario_family",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    return bundle


@pytest.fixture
def mock_config() -> dict:
    """Return a minimal test configuration dict."""
    return {
        "model": {
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "main",
            "tokenizer_id": "Qwen/Qwen3-4B-Instruct-2507",
        },
        "data": {
            "max_seq_length": 4096,
            "train_file": "data/prepared/tau-knowledge-v1/training/lora/train.jsonl",
            "validation_file": "data/prepared/tau-knowledge-v1/training/lora/validation.jsonl",
        },
        "training_args": {
            "output_dir": "checkpoints/test",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "learning_rate": 2e-4,
            "seed": 42,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
        "osft": {
            "unfreeze_rank_ratio": 0.25,
        },
        "harness": {
            "mode": "agent_rag",
            "limits": {
                "max_turns": 20,
                "max_tool_calls": 15,
            },
        },
        "retrieval": {
            "embedding": {"model_id": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
        },
    }
