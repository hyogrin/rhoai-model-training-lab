"""Data schemas for bundles, samples, provenance, and quality reports."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SampleType(str, Enum):
    POLICY_QA = "policy_qa"
    POLICY_APPLICATION = "policy_application"
    TOOL_SELECTION = "tool_selection"
    TRAJECTORY = "trajectory"
    CLARIFICATION = "clarification"
    EXCEPTION = "exception"


class ValidationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"
    REVIEW_REQUIRED = "review_required"


class Message(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="One of: system, user, assistant, tool")
    content: str | None = Field(None, description="Text content")
    tool_calls: list[ToolCall] | None = Field(None, description="Assistant tool calls")
    tool_call_id: str | None = Field(None, description="Tool response reference ID")
    name: str | None = Field(None, description="Tool name for tool role messages")


class ToolCall(BaseModel):
    """A tool call within an assistant message."""
    id: str
    type: str = "function"
    function: ToolCallFunction


class ToolCallFunction(BaseModel):
    """Function details for a tool call."""
    name: str
    arguments: str  # JSON-encoded arguments


class ToolSchema(BaseModel):
    """Tool/function schema definition."""
    type: str = "function"
    function: ToolFunctionDef


class ToolFunctionDef(BaseModel):
    """Function definition within a tool schema."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class CanonicalSample(BaseModel):
    """A single canonical training/validation example.

    Provenance fields are kept in sidecar metadata, not in model inputs.
    """
    sample_id: str = Field(..., description="Unique sample identifier, e.g. synthetic-policy-0001")
    messages: list[Message] = Field(..., description="Chat messages")
    tools: list[ToolSchema] | None = Field(None, description="Available tool schemas")
    source_doc_ids: list[str] | None = Field(None, description="Source document IDs")
    scenario_family: str | None = Field(None, description="Scenario family for split isolation")
    sample_type: SampleType = Field(..., description="Type of training example")
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING)


class ProvenanceRecord(BaseModel):
    """Provenance metadata for a canonical sample."""
    sample_id: str
    source_documents: list[str] = Field(default_factory=list)
    scenario_family: str = ""
    generator_id: str = ""
    generator_model: str = ""
    validator_id: str = ""
    validator_model: str = ""
    validation_outcome: ValidationStatus = ValidationStatus.PENDING
    rejection_reason: str | None = None
    generation_timestamp: str = ""
    validation_timestamp: str = ""


class SplitInfo(BaseModel):
    """Information about train/validation split."""
    method: str = "scenario_family"
    seed: int = 42
    train_ids: list[str] = Field(default_factory=list)
    validation_ids: list[str] = Field(default_factory=list)
    holdout_ids: list[str] = Field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    family_partition: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of scenario_family -> split assignment",
    )


class QualityReport(BaseModel):
    """Quality assessment of generated data."""
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    acceptance_rate: float = 0.0
    by_type: dict[str, TypeQualityStats] = Field(default_factory=dict)
    token_stats: TokenStats | None = None
    coverage: CoverageStats | None = None
    review_results: ReviewResults | None = None


class TypeQualityStats(BaseModel):
    """Quality stats for a specific sample type."""
    generated: int = 0
    accepted: int = 0
    rejected: int = 0
    acceptance_rate: float = 0.0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)


class TokenStats(BaseModel):
    """Token distribution statistics."""
    total_tokens: int = 0
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    p95_tokens: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0


class CoverageStats(BaseModel):
    """Coverage of KB documents and policy areas."""
    kb_documents_referenced: int = 0
    kb_documents_total: int = 0
    policy_areas_covered: list[str] = Field(default_factory=list)
    tool_schemas_covered: list[str] = Field(default_factory=list)


class ReviewResults(BaseModel):
    """Human/automated review results."""
    sample_size: int = 0
    reviewed: int = 0
    approved: int = 0
    flagged: int = 0
    review_notes: list[str] = Field(default_factory=list)


class BundleManifest(BaseModel):
    """Manifest for a prepared dataset bundle."""
    bundle_name: str
    bundle_version: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tau_version: str = ""
    tau_commit_sha: str = ""
    tau_task_revision: str = ""
    model_id: str = ""
    model_revision: str = ""
    tokenizer_id: str = ""
    chat_template_hash: str = ""
    source_hashes: dict[str, str] = Field(default_factory=dict)
    canonical_train_count: int = 0
    canonical_validation_count: int = 0
    split_policy: str = "scenario_family"
    sample_type_distribution: dict[str, int] = Field(default_factory=dict)
    lora_export_hash: str = ""
    osft_export_hash: str = ""
    bundle_hash: str = ""


class DatasetCard(BaseModel):
    """Dataset card for documentation."""
    name: str
    version: str
    description: str = ""
    generation_method: str = ""
    validation_method: str = ""
    split_methodology: str = ""
    limitations: list[str] = Field(default_factory=list)
    redistribution_conditions: list[str] = Field(default_factory=list)
    training_types: list[str] = Field(default_factory=list)
    model_profile: str = ""
    creation_date: str = ""


class BackendValidationResult(BaseModel):
    """Result of backend loader validation."""
    backend: str  # "lora" or "osft"
    loader_check: bool = False
    tokenizer_check: bool = False
    sample_count_match: bool = False
    chat_template_check: bool = False
    loss_mask_check: bool = False
    max_length_check: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    verified_scope: str = ""
