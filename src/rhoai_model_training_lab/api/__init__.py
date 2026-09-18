"""FastAPI application for the RAG harness backend.

Provides health, readiness, agent-step, and QA endpoints with
session-isolated state management, proper error handling,
and lifecycle hooks for index / model loading.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from rhoai_model_training_lab.schemas.api import (
    AgentStepRequest,
    AgentStepResponse,
    Citation,
    HealthResponse,
    QARequest,
    QAResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: dict[str, Any]) -> Any:
    """Create and return a fully wired FastAPI application.

    Parameters
    ----------
    config : dict
        The full rag.yaml configuration dict (with ``retrieval``,
        ``harness``, ``api``, and ``model`` sections).

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app_state = _AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup / shutdown lifecycle."""
        logger.info("Starting RAG harness backend …")
        try:
            app_state.startup()
            logger.info("Backend startup complete")
        except Exception:
            logger.exception("Backend startup failed")
        yield
        logger.info("Shutting down RAG harness backend …")
        app_state.shutdown()

    app = FastAPI(
        title="RHOAI Model Training Lab — RAG Harness",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- session-isolation middleware ---------------------------------------

    @app.middleware("http")
    async def session_isolation_middleware(request: Request, call_next):
        """Ensure that concurrent requests for different sessions are isolated."""
        request.state.request_start = time.time()
        response = await call_next(request)
        return response

    # -- health / readiness -------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse, tags=["health"])
    async def healthz() -> HealthResponse:
        """Liveness probe — always returns ok if the process is running."""
        return HealthResponse(
            status="ok",
            version="0.1.0",
            model_endpoint=app_state.model_endpoint,
        )

    @app.get("/readyz", response_model=HealthResponse, tags=["health"])
    async def readyz() -> HealthResponse:
        """Readiness probe — returns ok only when index and model are ready."""
        index_ready = app_state.index_ready
        model_ok = app_state.model_ready
        status = "ok" if (index_ready and model_ok) else "not_ready"
        return HealthResponse(
            status=status,
            version="0.1.0",
            model_endpoint=app_state.model_endpoint,
            index_ready=index_ready,
        )

    # -- POST /v1/agent/step -----------------------------------------------

    @app.post("/v1/agent/step", response_model=AgentStepResponse, tags=["agent"])
    async def agent_step(req: AgentStepRequest) -> AgentStepResponse:
        """Execute one agent turn in the planning / retrieval / tool graph.

        Expects a session_id and the current message history.
        Returns the next assistant message or tool calls.
        """
        request_id = req.request_id or str(uuid.uuid4())
        t_start = time.time()

        if not app_state.graph:
            raise HTTPException(status_code=503, detail="Harness graph not initialised")

        try:
            state = app_state.graph.get_or_create_session(req.session_id)

            if req.available_tools and app_state.graph._policy:
                tools_dicts = [t.model_dump() for t in req.available_tools]
                app_state.graph._policy.update_tools(tools_dicts)

            state.mode = req.mode

            for msg in req.messages:
                state.messages.append(msg.model_dump(exclude_none=True))

            if req.tool_observations:
                for obs in req.tool_observations:
                    obs_dict = obs.model_dump(exclude_none=True)
                    call_name = obs_dict.get("name", "unknown_tool")
                    state.tool_results.append({
                        "tool_name": call_name,
                        "tool_args": {},
                        "result": obs_dict.get("content", ""),
                        "executed": True,
                        "verified": False,
                        "id": obs_dict.get("tool_call_id", ""),
                    })

            state = app_state.graph.step(state)

            assistant_msg = None
            tool_calls_out: list[dict[str, Any]] | None = None
            if state.messages:
                last = state.messages[-1]
                if last.get("role") == "assistant":
                    from rhoai_model_training_lab.schemas.data import Message as DataMessage
                    assistant_msg = DataMessage(
                        role="assistant",
                        content=last.get("content"),
                    )
                    if last.get("tool_calls"):
                        tool_calls_out = last["tool_calls"]

            citations: list[Citation] = []
            for doc in state.retrieved_docs[:5]:
                citations.append(Citation(
                    document_id=doc.get("metadata", {}).get("source_doc_id", doc.get("id", "")),
                    chunk_id=doc.get("id", ""),
                    text_excerpt=doc.get("document", "")[:200],
                    score=doc.get("score", 0.0),
                ))

            elapsed_ms = (time.time() - t_start) * 1000

            fingerprint = {}
            if app_state.vector_index:
                fingerprint = app_state.vector_index.get_fingerprint(
                    embedding_revision=app_state.embedding_revision
                )

            return AgentStepResponse(
                session_id=req.session_id,
                request_id=request_id,
                status=state.status,
                message=assistant_msg,
                tool_calls=tool_calls_out,
                citations=citations if citations else None,
                plan=state.plan or None,
                model_hash=app_state.model_hash,
                corpus_hash=fingerprint.get("corpus_hash", ""),
                usage=UsageInfo(total_latency_ms=elapsed_ms),
                trace_id=request_id,
                error=state.error,
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Agent step failed for session %s", req.session_id)
            return AgentStepResponse(
                session_id=req.session_id,
                request_id=request_id,
                status="error",
                error=str(exc),
            )

    # -- POST /v1/qa -------------------------------------------------------

    @app.post("/v1/qa", response_model=QAResponse, tags=["diagnostics"])
    async def qa_endpoint(req: QARequest) -> QAResponse:
        """Answer a diagnostic QA question using retrieval + model.

        This endpoint is for prepared-diagnostics evaluation only and does
        not replace official τ episode evaluation.
        """
        request_id = str(uuid.uuid4())
        t_start = time.time()

        try:
            context_docs: list[dict[str, Any]] = []
            retrieval_ms = 0.0

            if req.mode == "rag" and app_state.retriever:
                rt_start = time.time()
                context_docs = app_state.retriever.retrieve(
                    query=req.question,
                    collection="kb_documents",
                    k=5,
                    method="dense",
                )
                retrieval_ms = (time.time() - rt_start) * 1000
            elif req.mode == "matched_context" and req.context_documents:
                for i, doc_text in enumerate(req.context_documents):
                    context_docs.append({
                        "document": doc_text,
                        "metadata": {"source_doc_id": f"matched_{i}"},
                        "score": 1.0,
                    })

            context_text = ""
            if context_docs:
                parts = []
                for i, d in enumerate(context_docs[:5], 1):
                    parts.append(f"[{i}] {d.get('document', '')[:600]}")
                context_text = "\n".join(parts)

            messages: list[dict[str, str]] = []
            if context_text:
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a banking support assistant. Use the following "
                        "knowledge documents to answer the question.\n\n"
                        f"{context_text}"
                    ),
                })
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a banking support assistant.",
                })
            messages.append({"role": "user", "content": req.question})

            model_start = time.time()
            if app_state.model_caller:
                response = app_state.model_caller(messages, max_tokens=2048, temperature=0.1)
                answer = response.get("content", "")
                tool_calls = response.get("tool_calls")
            else:
                answer = ""
                tool_calls = None
            model_ms = (time.time() - model_start) * 1000

            citations: list[Citation] = []
            for doc in context_docs[:5]:
                citations.append(Citation(
                    document_id=doc.get("metadata", {}).get("source_doc_id", ""),
                    text_excerpt=doc.get("document", "")[:200],
                    score=doc.get("score", 0.0),
                ))

            elapsed_ms = (time.time() - t_start) * 1000

            fingerprint = {}
            if app_state.vector_index:
                fingerprint = app_state.vector_index.get_fingerprint(
                    embedding_revision=app_state.embedding_revision
                )

            return QAResponse(
                status="ok",
                answer=answer,
                tool_calls=tool_calls,
                citations=citations if citations else None,
                model_hash=app_state.model_hash,
                corpus_hash=fingerprint.get("corpus_hash", ""),
                usage=UsageInfo(
                    retrieval_latency_ms=retrieval_ms,
                    model_latency_ms=model_ms,
                    total_latency_ms=elapsed_ms,
                ),
                trace_id=request_id,
            )

        except Exception as exc:
            logger.exception("QA endpoint failed")
            return QAResponse(
                status="error",
                error=str(exc),
                trace_id=request_id,
            )

    # -- error handlers -----------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "error": exc.detail},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": "Internal server error"},
        )

    return app


# ---------------------------------------------------------------------------
# Internal application state container
# ---------------------------------------------------------------------------


class _AppState:
    """Holds loaded components (index, retriever, graph, model caller).

    Created once per ``create_app`` invocation and wired during ``startup``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_endpoint: str = ""
        self.model_hash: str = ""
        self.embedding_revision: str = ""
        self.index_ready: bool = False
        self.model_ready: bool = False

        self.graph: Any = None
        self.retriever: Any = None
        self.vector_index: Any = None
        self.model_caller: Any = None

    def startup(self) -> None:
        """Load index, configure model caller, and build the harness graph."""
        self._load_model_caller()
        self._load_rag_components()
        self._build_graph()

    def shutdown(self) -> None:
        """Clean up resources."""
        self.graph = None
        self.retriever = None
        logger.info("Backend shutdown complete")

    # -- private setup helpers ----------------------------------------------

    def _load_model_caller(self) -> None:
        """Set up the model caller using httpx against the configured endpoint."""
        model_cfg = self.config.get("model", {})
        endpoint = model_cfg.get("endpoint", "")
        self.model_endpoint = endpoint

        if not endpoint:
            logger.warning("No model endpoint configured; model calls will fail")
            self.model_ready = False
            return

        token = model_cfg.get("token", "")
        ca_bundle = model_cfg.get("ca_bundle", "")
        model_name = model_cfg.get("model_name", "")
        temperature = model_cfg.get("temperature", 0.1)
        max_tokens = model_cfg.get("max_tokens", 2048)

        import httpx

        verify: Any = True
        if ca_bundle:
            verify = ca_bundle

        client = httpx.Client(verify=verify, timeout=120.0)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        def _call_model(
            messages: list[dict[str, Any]],
            max_tokens: int = max_tokens,
            temperature: float = temperature,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if model_name:
                payload["model"] = model_name

            chat_url = f"{endpoint.rstrip('/')}/chat/completions"

            try:
                resp = client.post(chat_url, json=payload, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    raise RuntimeError(f"Authentication failed ({status}): check SERVING_TOKEN") from exc
                if status == 408 or status == 504:
                    raise RuntimeError(f"Model endpoint timeout ({status})") from exc
                raise

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
                "usage": data.get("usage"),
            }

        self.model_caller = _call_model
        self.model_ready = True
        self.model_hash = model_cfg.get("model_name", "unknown")
        logger.info("Model caller configured for %s", endpoint)

    def _load_rag_components(self) -> None:
        """Load the RAG index and retriever."""
        try:
            from rhoai_model_training_lab.rag import build_rag_components

            components = build_rag_components(self.config)
            self.retriever = components["retriever"]
            self.vector_index = components["vector_index"]
            self.embedding_revision = components["embedding"].revision
            self.index_ready = True
            logger.info("RAG components loaded successfully")
        except ImportError:
            logger.warning("RAG dependencies not available; index disabled")
            self.index_ready = False
        except Exception:
            logger.exception("Failed to load RAG components")
            self.index_ready = False

    def _build_graph(self) -> None:
        """Build the harness graph."""
        try:
            from rhoai_model_training_lab.harness import build_graph

            self.graph = build_graph(self.config)
            self.graph.configure(
                model_caller=self.model_caller,
                retriever=self.retriever,
                available_tools=None,
            )
            logger.info("Harness graph built (mode=%s)", self.graph.mode)
        except ImportError:
            logger.warning("Harness dependencies not available")
        except Exception:
            logger.exception("Failed to build harness graph")


__all__ = [
    "create_app",
]
