"""Evaluation configuration and result schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalRunConfig(BaseModel):
    """Configuration for a single evaluation run."""
    track: str  # prepared_diagnostics | tau_episodes | retention
    variant: str = ""  # e.g. base_no_knowledge, lora_rag, etc.
    model_endpoint: str = ""
    model_hash: str = ""
    data_revision: str = ""
    bundle_id: str = ""
    config_hash: str = ""

    # τ episodes specific
    task_ids: list[str] | None = None
    trials: int = 3
    limit: int | None = None
    knowledge_access: str = "no_knowledge"
    mode: str = "agent_rag"
    use_examples: bool = False

    # Grading
    grader_version: str = ""
    scorer_revision: str = ""
    simulator_model: str = ""
    simulator_revision: str = ""
    simulator_seed: int | None = None

    # Budgets
    max_turns: int = 20
    max_tool_calls: int = 15
    max_tokens: int = 8192
    max_wall_time_seconds: int = 300

    seed: int = 42
    no_mlflow: bool = False


class TaskResult(BaseModel):
    """Result for a single evaluation task."""
    task_id: str
    trial: int = 0
    success: bool = False
    reward: float = 0.0
    actions: list[str] = Field(default_factory=list)
    expected_actions: list[str] | None = None  # Only for evaluator, never sent to model
    turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    retrieval_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    error: str | None = None
    error_type: str | None = None  # timeout | budget_exceeded | invalid_tool | api_error
    trace_path: str = ""


class EvalMetrics(BaseModel):
    """Aggregated evaluation metrics."""
    track: str
    variant: str = ""
    total_tasks: int = 0
    attempted: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    unsupported: int = 0
    timed_out: int = 0
    budget_exceeded: int = 0

    # Official τ metrics
    task_success_rate: float | None = None
    pass_k: float | None = None  # pass^k (consistent success)
    pass_k_k: int | None = None
    action_recall: float | None = None

    # Diagnostic metrics
    answer_accuracy: float | None = None
    tool_name_accuracy: float | None = None
    tool_args_accuracy: float | None = None
    grounding_score: float | None = None
    condition_accuracy: float | None = None

    # Confidence intervals
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_method: str = "paired_bootstrap"
    ci_alpha: float = 0.05

    # Latency/cost
    mean_turns: float = 0.0
    mean_tool_calls: float = 0.0
    mean_tokens: float = 0.0
    mean_wall_time_seconds: float = 0.0
    mean_retrieval_latency_ms: float = 0.0
    mean_model_latency_ms: float = 0.0

    # Failures
    error_counts: dict[str, int] = Field(default_factory=dict)


class RetentionResult(BaseModel):
    """Retention benchmark result."""
    benchmark: str  # arc_challenge, ifeval
    variant: str  # base, lora, osft
    sample_size: int = 0
    accuracy: float = 0.0
    base_accuracy: float = 0.0
    retention_delta_pp: float = 0.0  # 100 * (adapted - base)


class EvalResult(BaseModel):
    """Complete evaluation result for MLflow logging."""
    run_id: str = ""
    track: str = ""
    variant: str = ""
    config: EvalRunConfig | None = None
    metrics: EvalMetrics | None = None
    task_results: list[TaskResult] = Field(default_factory=list)
    retention_results: list[RetentionResult] = Field(default_factory=list)

    # Lineage
    data_run_id: str = ""
    training_run_id: str = ""
    bundle_id: str = ""
    bundle_hash: str = ""
    model_hash: str = ""
    corpus_hash: str = ""
    grader_version: str = ""
    scorer_revision: str = ""

    # Status
    execution_status: str = "not_run"  # not_run | running | completed | failed | blocked
    protocol_deviations: list[str] = Field(default_factory=list)
    verification_status: str = "unverified"  # unverified | fixture_verified | live_verified


class ComparisonRow(BaseModel):
    """A row in the experiment comparison table."""
    variant: str
    model: str
    knowledge_access: str = ""
    mode: str = ""
    task_success_rate: float | None = None
    pass_k: float | None = None
    diagnostic_accuracy: float | None = None
    retention_delta_pp: float | None = None
    mean_turns: float | None = None
    mean_tokens: float | None = None
    mean_wall_time: float | None = None
    failures: int = 0
    notes: str = ""
