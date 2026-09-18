"""API request/response schemas for the RAG harness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rhoai_model_training_lab.schemas.data import Message, ToolSchema


class HealthResponse(BaseModel):
    """GET /healthz and /readyz response."""
    status: str = "ok"
    version: str = ""
    model_endpoint: str = ""
    index_ready: bool = False


class AgentStepRequest(BaseModel):
    """POST /v1/agent/step request body."""
    session_id: str
    request_id: str = ""
    messages: list[Message] = Field(default_factory=list)
    tool_observations: list[Message] | None = None
    available_tools: list[ToolSchema] | None = None
    mode: str = "agent_rag"  # simple_rag | agent_rag
    use_examples: bool = False
    knowledge_access: str = "rag"  # no_knowledge | rag | full_kb | golden_retrieval


class AgentStepResponse(BaseModel):
    """POST /v1/agent/step response body."""
    session_id: str
    request_id: str = ""
    status: str = "ok"  # ok | error | budget_exceeded | terminated
    message: Message | None = None
    tool_calls: list[dict[str, Any]] | None = None
    citations: list[Citation] | None = None
    plan: str | None = None
    model_hash: str = ""
    corpus_hash: str = ""
    usage: UsageInfo | None = None
    trace_id: str = ""
    error: str | None = None


class Citation(BaseModel):
    """A citation reference."""
    document_id: str
    chunk_id: str = ""
    text_excerpt: str = ""
    score: float = 0.0


class UsageInfo(BaseModel):
    """Token and timing usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retrieval_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class QARequest(BaseModel):
    """POST /v1/qa request body for diagnostics."""
    question: str
    context_documents: list[str] | None = None
    mode: str = "rag"  # rag | matched_context
    model_variant: str = "base"  # base | lora | osft


class QAResponse(BaseModel):
    """POST /v1/qa response body."""
    status: str = "ok"
    answer: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    citations: list[Citation] | None = None
    model_hash: str = ""
    corpus_hash: str = ""
    usage: UsageInfo | None = None
    trace_id: str = ""
    error: str | None = None
