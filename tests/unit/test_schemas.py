"""Tests for all Pydantic schemas in rhoai_model_training_lab.schemas."""

from __future__ import annotations

import json

import pytest

from rhoai_model_training_lab.schemas.data import (
    BundleManifest,
    CanonicalSample,
    Message,
    SampleType,
    SplitInfo,
    ToolCall,
    ToolCallFunction,
    ToolSchema,
    ToolFunctionDef,
    ValidationStatus,
)
from rhoai_model_training_lab.schemas.training import (
    LoRAConfig,
    OSFTConfig,
    TrainingResult,
)
from rhoai_model_training_lab.schemas.evaluation import (
    EvalMetrics,
    EvalResult,
    EvalRunConfig,
    RetentionResult,
    TaskResult,
)
from rhoai_model_training_lab.schemas.api import (
    AgentStepRequest,
    AgentStepResponse,
    HealthResponse,
    QARequest,
    QAResponse,
)


class TestCanonicalSample:
    """Test CanonicalSample creation, serialisation, and defaults."""

    def test_create_minimal(self, sample_canonical):
        assert sample_canonical.sample_id == "test-policy-0001"
        assert sample_canonical.sample_type == SampleType.POLICY_QA
        assert len(sample_canonical.messages) == 3

    def test_serialization_roundtrip(self, sample_canonical):
        data = sample_canonical.model_dump()
        restored = CanonicalSample(**data)
        assert restored.sample_id == sample_canonical.sample_id
        assert restored.messages[0].content == sample_canonical.messages[0].content
        assert restored.sample_type == sample_canonical.sample_type

    def test_json_roundtrip(self, sample_canonical):
        json_str = sample_canonical.model_dump_json()
        restored = CanonicalSample.model_validate_json(json_str)
        assert restored == sample_canonical

    def test_scenario_family_preserved(self, sample_canonical):
        assert sample_canonical.scenario_family == "savings_interest"

    def test_validation_status_default(self):
        sample = CanonicalSample(
            sample_id="test-001",
            messages=[Message(role="user", content="hello")],
            sample_type=SampleType.POLICY_QA,
        )
        assert sample.validation_status == ValidationStatus.PENDING

    def test_all_sample_types(self):
        for st in SampleType:
            sample = CanonicalSample(
                sample_id=f"test-{st.value}",
                messages=[Message(role="user", content="test")],
                sample_type=st,
            )
            assert sample.sample_type == st


class TestBundleManifest:
    """Test BundleManifest validation and defaults."""

    def test_create_with_required_fields(self):
        manifest = BundleManifest(
            bundle_name="tau-knowledge",
            bundle_version="v1",
        )
        assert manifest.bundle_name == "tau-knowledge"
        assert manifest.bundle_version == "v1"
        assert manifest.canonical_train_count == 0

    def test_created_at_auto_populated(self):
        manifest = BundleManifest(
            bundle_name="test",
            bundle_version="v1",
        )
        assert manifest.created_at  # non-empty

    def test_sample_type_distribution(self):
        manifest = BundleManifest(
            bundle_name="test",
            bundle_version="v1",
            sample_type_distribution={"policy_qa": 50, "trajectory": 30},
        )
        assert manifest.sample_type_distribution["policy_qa"] == 50
        assert manifest.sample_type_distribution["trajectory"] == 30

    def test_model_id_default(self):
        manifest = BundleManifest(bundle_name="test", bundle_version="v1")
        assert manifest.model_id == ""

    def test_full_manifest(self):
        manifest = BundleManifest(
            bundle_name="tau-knowledge",
            bundle_version="v1",
            tau_version="1.0.1",
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            canonical_train_count=200,
            canonical_validation_count=30,
            split_policy="scenario_family",
            bundle_hash="abc123",
        )
        data = manifest.model_dump()
        assert data["canonical_train_count"] == 200
        assert data["split_policy"] == "scenario_family"


class TestMessage:
    """Test Message with and without tool_calls."""

    def test_simple_message(self, sample_message):
        assert sample_message.role == "assistant"
        assert sample_message.content is not None
        assert sample_message.tool_calls is None

    def test_message_with_tool_calls(self, sample_tool_call_message):
        msg = sample_tool_call_message
        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.id == "call_001"
        assert tc.function.name == "get_account_balance"
        parsed_args = json.loads(tc.function.arguments)
        assert parsed_args["account_id"] == "ACC-12345"

    def test_tool_role_message(self):
        msg = Message(
            role="tool",
            content='{"balance": 1234.56}',
            tool_call_id="call_001",
            name="get_account_balance",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_001"
        assert msg.name == "get_account_balance"

    def test_system_message(self):
        msg = Message(role="system", content="You are a banking assistant.")
        assert msg.role == "system"
        assert msg.tool_calls is None


class TestEvalRunConfig:
    """Test EvalRunConfig defaults and overrides."""

    def test_defaults(self):
        cfg = EvalRunConfig(track="tau_episodes")
        assert cfg.track == "tau_episodes"
        assert cfg.trials == 3
        assert cfg.max_turns == 20
        assert cfg.max_tool_calls == 15
        assert cfg.max_tokens == 8192
        assert cfg.max_wall_time_seconds == 300
        assert cfg.seed == 42
        assert cfg.no_mlflow is False
        assert cfg.knowledge_access == "no_knowledge"

    def test_custom_values(self):
        cfg = EvalRunConfig(
            track="prepared_diagnostics",
            variant="lora_rag",
            trials=5,
            limit=10,
            knowledge_access="rag",
        )
        assert cfg.variant == "lora_rag"
        assert cfg.trials == 5
        assert cfg.limit == 10
        assert cfg.knowledge_access == "rag"


class TestTrainingResult:
    """Test TrainingResult fields and serialisation."""

    def test_required_fields(self):
        result = TrainingResult(method="lora")
        assert result.method == "lora"
        assert result.model_id == ""
        assert result.final_train_loss == 0.0
        assert result.seed == 42

    def test_full_result(self):
        result = TrainingResult(
            method="osft",
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            train_samples=200,
            validation_samples=30,
            total_steps=750,
            num_epochs=3,
            final_train_loss=0.45,
            final_eval_loss=0.52,
            wall_time_seconds=3600.0,
            peak_vram_gb=22.5,
            gpu_name="NVIDIA A10G",
        )
        assert result.method == "osft"
        assert result.train_samples == 200
        assert result.final_eval_loss == 0.52
        assert result.gpu_name == "NVIDIA A10G"

    def test_serialization_roundtrip(self):
        result = TrainingResult(
            method="lora",
            model_id="test",
            final_train_loss=0.3,
        )
        data = result.model_dump()
        restored = TrainingResult(**data)
        assert restored.method == result.method
        assert restored.final_train_loss == result.final_train_loss


class TestLoRAConfig:
    """Test LoRAConfig defaults."""

    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.model_id == "Qwen/Qwen3-4B-Instruct-2507"
        assert cfg.loss_masking == "assistant_only"
        assert cfg.max_seq_length == 4096


class TestOSFTConfig:
    """Test OSFTConfig defaults."""

    def test_defaults(self):
        cfg = OSFTConfig()
        assert cfg.unfreeze_rank_ratio == 0.25
        assert cfg.model_id == "Qwen/Qwen3-4B-Instruct-2507"
        assert cfg.loss_masking == "assistant_only"


class TestAPISchemas:
    """Test API request/response schemas."""

    def test_health_response_defaults(self):
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.index_ready is False

    def test_agent_step_request(self):
        req = AgentStepRequest(
            session_id="sess-001",
            messages=[Message(role="user", content="hello")],
        )
        assert req.session_id == "sess-001"
        assert req.mode == "agent_rag"

    def test_qa_request(self):
        req = QARequest(question="What is the interest rate?")
        assert req.mode == "rag"
        assert req.model_variant == "base"

    def test_qa_response(self):
        resp = QAResponse(answer="4.5% APY")
        assert resp.status == "ok"
        assert resp.answer == "4.5% APY"


class TestRetentionResult:
    """Test RetentionResult delta computation fields."""

    def test_positive_retention(self):
        r = RetentionResult(
            benchmark="arc_challenge",
            variant="lora",
            sample_size=100,
            accuracy=0.75,
            base_accuracy=0.70,
            retention_delta_pp=5.0,
        )
        assert r.retention_delta_pp == 5.0

    def test_negative_retention(self):
        r = RetentionResult(
            benchmark="arc_challenge",
            variant="osft",
            accuracy=0.65,
            base_accuracy=0.70,
            retention_delta_pp=-5.0,
        )
        assert r.retention_delta_pp == -5.0


class TestEvalMetrics:
    """Test EvalMetrics aggregated fields."""

    def test_pass_k_fields(self):
        m = EvalMetrics(
            track="tau_episodes",
            pass_k=0.6,
            pass_k_k=3,
        )
        assert m.pass_k == 0.6
        assert m.pass_k_k == 3

    def test_ci_fields(self):
        m = EvalMetrics(
            track="tau_episodes",
            ci_lower=0.45,
            ci_upper=0.75,
            ci_method="paired_bootstrap",
            ci_alpha=0.05,
        )
        assert m.ci_lower == 0.45
        assert m.ci_upper == 0.75
