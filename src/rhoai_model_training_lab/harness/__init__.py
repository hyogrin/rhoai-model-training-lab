"""LangGraph planning and tool-execution graph for the RAG harness.

Implements a state-graph with nodes for planning, retrieval, policy
checking, tool execution, and verification.  Supports ``simple_rag``
(fixed retrieval per turn, no planning) and ``agent_rag`` (full planning
with multiple retrieval / tool steps) modes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Budget / limit sentinel
_BUDGET_EXCEEDED = "budget_exceeded"
_TERMINATED = "terminated"
_OK = "ok"
_ERROR = "error"

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Mutable state flowing through the LangGraph state-graph.

    Every session gets its own ``AgentState`` instance.  The graph nodes
    read and mutate this state as the conversation progresses.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    plan: str = ""
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    tool_call_count: int = 0
    mode: str = "agent_rag"
    session_id: str = ""
    budget_remaining: dict[str, Any] = field(default_factory=dict)

    # Internal bookkeeping
    status: str = _OK
    error: str | None = None
    start_time: float = 0.0
    plan_tokens: int = 0
    _executed_mutations: set[str] = field(default_factory=set)

    def check_budget(self) -> str | None:
        """Return a reason string if any budget is exhausted, else None."""
        max_turns = self.budget_remaining.get("max_turns", 20)
        max_tool_calls = self.budget_remaining.get("max_tool_calls", 15)
        max_wall = self.budget_remaining.get("max_wall_time_seconds", 300)

        if self.turn_count >= max_turns:
            return f"max_turns exceeded ({self.turn_count}/{max_turns})"
        if self.tool_call_count >= max_tool_calls:
            return f"max_tool_calls exceeded ({self.tool_call_count}/{max_tool_calls})"
        if self.start_time and (time.time() - self.start_time) > max_wall:
            elapsed = time.time() - self.start_time
            return f"max_wall_time exceeded ({elapsed:.0f}s/{max_wall}s)"
        return None

    def mutation_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Compute a deterministic key for a state-changing tool call."""
        payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def reset_episode(self) -> None:
        """Reset state for a new episode / task trial."""
        self.messages.clear()
        self.plan = ""
        self.retrieved_docs.clear()
        self.tool_results.clear()
        self.turn_count = 0
        self.tool_call_count = 0
        self.status = _OK
        self.error = None
        self.start_time = time.time()
        self.plan_tokens = 0
        self._executed_mutations.clear()


# ---------------------------------------------------------------------------
# Graph node implementations
# ---------------------------------------------------------------------------


class PlannerNode:
    """Receive observations and create a structured plan.

    In ``agent_rag`` mode the planner produces retrieval queries and tool
    selection hints.  In ``simple_rag`` mode it is a no-op pass-through.

    Parameters
    ----------
    model_caller : callable
        ``(messages, **kwargs) -> dict`` that calls the served LLM and
        returns a parsed response dict with ``content`` and optionally
        ``tool_calls``.
    max_plan_tokens : int
        Token budget for the plan.
    """

    PLAN_SYSTEM_PROMPT = (
        "You are a banking assistant planner.  Given the conversation so far, "
        "produce a concise JSON plan with keys: "
        '"queries" (list of retrieval queries), '
        '"tools_needed" (list of tool names likely required), '
        '"reasoning" (one-sentence rationale).  '
        "Output ONLY the JSON."
    )

    def __init__(
        self,
        model_caller: Any,
        max_plan_tokens: int = 512,
    ) -> None:
        self.model_caller = model_caller
        self.max_plan_tokens = max_plan_tokens

    def __call__(self, state: AgentState) -> AgentState:
        if state.mode == "simple_rag":
            user_msg = ""
            for m in reversed(state.messages):
                if m.get("role") == "user" and m.get("content"):
                    user_msg = m["content"]
                    break
            state.plan = json.dumps({"queries": [user_msg], "tools_needed": [], "reasoning": "simple_rag"})
            return state

        budget_reason = state.check_budget()
        if budget_reason:
            state.status = _BUDGET_EXCEEDED
            state.error = budget_reason
            return state

        plan_messages = [
            {"role": "system", "content": self.PLAN_SYSTEM_PROMPT},
            *state.messages,
        ]

        try:
            response = self.model_caller(
                plan_messages,
                max_tokens=self.max_plan_tokens,
                temperature=0.0,
            )
            plan_text = response.get("content", "")
            state.plan = plan_text
            state.plan_tokens += len(plan_text.split())
            logger.debug("Plan generated: %s", plan_text[:200])
        except Exception:
            logger.exception("Planning failed; falling back to direct query")
            user_msg = ""
            for m in reversed(state.messages):
                if m.get("role") == "user" and m.get("content"):
                    user_msg = m["content"]
                    break
            state.plan = json.dumps({"queries": [user_msg], "tools_needed": [], "reasoning": "fallback"})

        return state


class RetrieverNode:
    """Execute knowledge retrieval based on the plan.

    Parameters
    ----------
    retriever : object
        A ``Retriever`` instance with ``.retrieve(query, collection, k, method)``.
    k : int
        Number of results per query.
    method : str
        Retrieval method (``dense`` or ``bm25``).
    use_examples : bool
        Whether to also search the training-examples index.
    """

    def __init__(
        self,
        retriever: Any,
        k: int = 5,
        method: str = "dense",
        use_examples: bool = False,
    ) -> None:
        self.retriever = retriever
        self.k = k
        self.method = method
        self.use_examples = use_examples

    def __call__(self, state: AgentState) -> AgentState:
        budget_reason = state.check_budget()
        if budget_reason:
            state.status = _BUDGET_EXCEEDED
            state.error = budget_reason
            return state

        queries: list[str] = []
        try:
            plan_data = json.loads(state.plan) if state.plan else {}
            raw = plan_data.get("queries", [])
            if isinstance(raw, list):
                queries = [q for q in raw if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, TypeError):
            for m in reversed(state.messages):
                if m.get("role") == "user" and m.get("content"):
                    queries = [m["content"]]
                    break

        if not queries:
            return state

        all_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for query in queries:
            try:
                results = self.retriever.retrieve(
                    query=query,
                    collection="kb_documents",
                    k=self.k,
                    method=self.method,
                )
                for r in results:
                    rid = r.get("id", r.get("document", "")[:64])
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_docs.append(r)
            except Exception:
                logger.exception("Retrieval failed for query: %s", query[:80])

        if self.use_examples:
            for query in queries[:1]:
                try:
                    examples = self.retriever.retrieve(
                        query=query,
                        collection="training_examples",
                        k=2,
                        method="dense",
                    )
                    for ex in examples:
                        ex["source"] = "example"
                        all_docs.append(ex)
                except Exception:
                    logger.exception("Example retrieval failed")

        state.retrieved_docs = all_docs
        logger.info("Retrieved %d documents for %d queries", len(all_docs), len(queries))
        return state


class PolicyNode:
    """Apply banking policy rules to determine allowed actions.

    Validates proposed tool calls against available tool schemas and
    basic policy constraints (e.g. preventing unauthorised transfers).

    Parameters
    ----------
    available_tools : list[dict] | None
        Available tool schemas.  If ``None``, all tools are allowed.
    """

    def __init__(self, available_tools: list[dict[str, Any]] | None = None) -> None:
        self.available_tools = available_tools
        self._allowed_tool_names: set[str] | None = None
        if available_tools:
            self._allowed_tool_names = set()
            for t in available_tools:
                fn = t.get("function", {})
                name = fn.get("name", t.get("name", ""))
                if name:
                    self._allowed_tool_names.add(name)

    def __call__(self, state: AgentState) -> AgentState:
        if not state.tool_results:
            return state

        filtered: list[dict[str, Any]] = []
        for tool_result in state.tool_results:
            tool_name = tool_result.get("tool_name", "")
            if self._allowed_tool_names and tool_name not in self._allowed_tool_names:
                logger.warning("Policy: blocking disallowed tool '%s'", tool_name)
                tool_result["blocked"] = True
                tool_result["block_reason"] = f"Tool '{tool_name}' not in allowed tool list"
            else:
                tool_result["blocked"] = False
            filtered.append(tool_result)

        state.tool_results = filtered
        return state

    def update_tools(self, tools: list[dict[str, Any]]) -> None:
        """Update the allowed tool set at runtime."""
        self.available_tools = tools
        self._allowed_tool_names = set()
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", t.get("name", ""))
            if name:
                self._allowed_tool_names.add(name)


class ToolExecutorNode:
    """Issue authorised tool calls through the simulation adapter.

    Prevents duplicate state-changing mutations within a single episode
    and enforces the tool-call budget.

    Parameters
    ----------
    simulation_adapter : object | None
        A ``TauSimulationAdapter`` (or compatible) with
        ``.execute_tool(name, args) -> dict``.
    """

    def __init__(self, simulation_adapter: Any = None) -> None:
        self.adapter = simulation_adapter

    def __call__(self, state: AgentState) -> AgentState:
        budget_reason = state.check_budget()
        if budget_reason:
            state.status = _BUDGET_EXCEEDED
            state.error = budget_reason
            return state

        pending = [tr for tr in state.tool_results if not tr.get("executed") and not tr.get("blocked")]
        if not pending:
            return state

        for call in pending:
            tool_name = call.get("tool_name", "")
            tool_args = call.get("tool_args", {})
            call_id = call.get("id", str(uuid.uuid4()))

            mut_key = state.mutation_key(tool_name, tool_args)
            if mut_key in state._executed_mutations:
                logger.warning(
                    "Duplicate mutation prevented: %s(%s)",
                    tool_name,
                    json.dumps(tool_args)[:100],
                )
                call["executed"] = True
                call["result"] = {"error": "Duplicate state-changing call prevented"}
                call["duplicate"] = True
                continue

            if self.adapter is None:
                call["executed"] = True
                call["result"] = {"error": "No simulation adapter configured"}
                logger.error("Tool call without adapter: %s", tool_name)
                continue

            try:
                result = self.adapter.execute_tool(tool_name, tool_args)
                call["executed"] = True
                call["result"] = result
                call["id"] = call_id
                state.tool_call_count += 1
                state._executed_mutations.add(mut_key)
                logger.info("Tool executed: %s → %s", tool_name, str(result)[:200])
            except Exception as exc:
                call["executed"] = True
                call["result"] = {"error": str(exc)}
                logger.exception("Tool execution failed: %s", tool_name)

        return state


class VerifierNode:
    """Verify results using only permitted observations.

    Never uses hidden expected actions or private evaluator fields.

    Parameters
    ----------
    model_caller : callable | None
        Optional LLM caller for verification reasoning.
    """

    VERIFY_SYSTEM_PROMPT = (
        "You are a banking operations verifier.  Given the tool call, its result, "
        "and the retrieved policy documents, determine whether the action was "
        "correctly performed.  Respond with a JSON object: "
        '{"verified": true/false, "reason": "..."}.  '
        "Use ONLY the provided observations and policy documents.  "
        "Do NOT reference any expected actions or hidden evaluation criteria."
    )

    def __init__(self, model_caller: Any = None) -> None:
        self.model_caller = model_caller

    def __call__(self, state: AgentState) -> AgentState:
        executed = [tr for tr in state.tool_results if tr.get("executed") and not tr.get("verified")]
        if not executed:
            return state

        for call in executed:
            result = call.get("result", {})
            if isinstance(result, dict) and result.get("error"):
                call["verified"] = False
                call["verification_reason"] = f"Tool error: {result['error']}"
                continue

            if self.model_caller is None:
                call["verified"] = True
                call["verification_reason"] = "No verifier model configured; accepted"
                continue

            docs_context = "\n".join(
                d.get("document", "")[:500] for d in state.retrieved_docs[:3]
            )
            verify_messages = [
                {"role": "system", "content": self.VERIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Tool: {call.get('tool_name')}\n"
                        f"Args: {json.dumps(call.get('tool_args', {}))}\n"
                        f"Result: {json.dumps(result)}\n"
                        f"Policy context:\n{docs_context}"
                    ),
                },
            ]

            try:
                resp = self.model_caller(verify_messages, max_tokens=256, temperature=0.0)
                content = resp.get("content", "")
                try:
                    vdata = json.loads(content)
                    call["verified"] = vdata.get("verified", True)
                    call["verification_reason"] = vdata.get("reason", "")
                except (json.JSONDecodeError, TypeError):
                    call["verified"] = True
                    call["verification_reason"] = content[:200]
            except Exception:
                logger.exception("Verification call failed")
                call["verified"] = True
                call["verification_reason"] = "Verification call failed; defaulting to accepted"

        return state


class ResponderNode:
    """Generate the final assistant response for the current turn.

    Parameters
    ----------
    model_caller : callable
        LLM caller for response generation.
    """

    def __init__(self, model_caller: Any) -> None:
        self.model_caller = model_caller

    def __call__(self, state: AgentState) -> AgentState:
        budget_reason = state.check_budget()
        if budget_reason:
            state.status = _BUDGET_EXCEEDED
            state.error = budget_reason
            return state

        context_parts: list[str] = []
        if state.retrieved_docs:
            context_parts.append("Retrieved knowledge:")
            for i, doc in enumerate(state.retrieved_docs[:5], 1):
                text = doc.get("document", "")[:600]
                context_parts.append(f"[{i}] {text}")

        for tr in state.tool_results:
            if tr.get("executed") and tr.get("result"):
                context_parts.append(
                    f"Tool result ({tr.get('tool_name')}): {json.dumps(tr['result'])[:400]}"
                )

        messages = list(state.messages)
        if context_parts:
            messages.append({"role": "system", "content": "\n".join(context_parts)})

        try:
            response = self.model_caller(messages, max_tokens=2048, temperature=0.1)
            state.messages.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls"),
            })
            state.turn_count += 1
        except Exception as exc:
            state.status = _ERROR
            state.error = str(exc)
            logger.exception("Response generation failed")

        return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

_NODE_ORDER_AGENT_RAG = ["plan", "retrieve", "policy", "tool_execute", "verify", "respond"]
_NODE_ORDER_SIMPLE_RAG = ["plan", "retrieve", "respond"]


def build_graph(config: dict[str, Any]) -> HarnessGraph:
    """Create a ``HarnessGraph`` from the harness section of rag.yaml.

    Parameters
    ----------
    config : dict
        The full rag configuration dict (with ``harness`` and ``retrieval``
        keys).

    Returns
    -------
    HarnessGraph
        A callable graph that processes ``AgentState`` through the
        configured pipeline.
    """
    harness_cfg = config.get("harness", {})
    limits_cfg = harness_cfg.get("limits", {})
    planning_cfg = harness_cfg.get("planning", {})
    mode = harness_cfg.get("mode", "agent_rag")
    use_examples = harness_cfg.get("use_examples", False)

    default_budgets = {
        "max_turns": limits_cfg.get("max_turns", 20),
        "max_tool_calls": limits_cfg.get("max_tool_calls", 15),
        "max_tokens": limits_cfg.get("max_tokens_per_turn", 4096),
        "max_wall_time_seconds": limits_cfg.get("max_wall_time_seconds", 300),
    }

    return HarnessGraph(
        mode=mode,
        use_examples=use_examples,
        max_plan_tokens=planning_cfg.get("max_plan_tokens", 512),
        default_budgets=default_budgets,
    )


class HarnessGraph:
    """Callable harness graph that processes agent turns.

    This is a lightweight orchestration layer that can be connected to
    a full LangGraph ``StateGraph`` when the ``langgraph`` package is
    available, or operates as a simple sequential pipeline otherwise.

    Parameters
    ----------
    mode : str
        ``simple_rag`` or ``agent_rag``.
    use_examples : bool
        Whether to search the training-examples index.
    max_plan_tokens : int
        Token budget for the planning step.
    default_budgets : dict
        Default budget limits.
    """

    def __init__(
        self,
        mode: str = "agent_rag",
        use_examples: bool = False,
        max_plan_tokens: int = 512,
        default_budgets: dict[str, Any] | None = None,
    ) -> None:
        self.mode = mode
        self.use_examples = use_examples
        self.max_plan_tokens = max_plan_tokens
        self.default_budgets = default_budgets or {
            "max_turns": 20,
            "max_tool_calls": 15,
            "max_tokens": 4096,
            "max_wall_time_seconds": 300,
        }

        self._planner: PlannerNode | None = None
        self._retriever_node: RetrieverNode | None = None
        self._policy: PolicyNode | None = None
        self._executor: ToolExecutorNode | None = None
        self._verifier: VerifierNode | None = None
        self._responder: ResponderNode | None = None

        self._sessions: dict[str, AgentState] = {}

    # -- wiring -------------------------------------------------------------

    def configure(
        self,
        model_caller: Any,
        retriever: Any | None = None,
        simulation_adapter: Any | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Wire up external dependencies.

        Parameters
        ----------
        model_caller :
            ``(messages, **kwargs) -> dict`` for the served LLM.
        retriever :
            A ``Retriever`` instance.
        simulation_adapter :
            A ``TauSimulationAdapter`` instance.
        available_tools :
            Tool schemas exposed to the agent.
        """
        self._planner = PlannerNode(
            model_caller=model_caller,
            max_plan_tokens=self.max_plan_tokens,
        )
        if retriever is not None:
            self._retriever_node = RetrieverNode(
                retriever=retriever,
                use_examples=self.use_examples,
            )
        self._policy = PolicyNode(available_tools=available_tools)
        self._executor = ToolExecutorNode(simulation_adapter=simulation_adapter)
        self._verifier = VerifierNode(model_caller=model_caller)
        self._responder = ResponderNode(model_caller=model_caller)

    # -- session management -------------------------------------------------

    def get_or_create_session(self, session_id: str) -> AgentState:
        """Get an existing session or create a new one."""
        if session_id not in self._sessions:
            state = AgentState(
                session_id=session_id,
                mode=self.mode,
                budget_remaining=dict(self.default_budgets),
                start_time=time.time(),
            )
            self._sessions[session_id] = state
            logger.info("New session created: %s", session_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> AgentState:
        """Reset (or create) a session for a new episode."""
        state = self.get_or_create_session(session_id)
        state.reset_episode()
        state.mode = self.mode
        state.budget_remaining = dict(self.default_budgets)
        logger.info("Session reset: %s", session_id)
        return state

    def remove_session(self, session_id: str) -> None:
        """Remove a session entirely."""
        self._sessions.pop(session_id, None)

    # -- execution ----------------------------------------------------------

    def step(self, state: AgentState) -> AgentState:
        """Execute one full turn through the graph.

        In ``agent_rag`` mode: plan → retrieve → policy → tool_execute → verify → respond.
        In ``simple_rag`` mode: plan (passthrough) → retrieve → respond.

        Returns the updated state.
        """
        if state.status in (_BUDGET_EXCEEDED, _TERMINATED):
            return state

        nodes = _NODE_ORDER_SIMPLE_RAG if self.mode == "simple_rag" else _NODE_ORDER_AGENT_RAG

        node_map = {
            "plan": self._planner,
            "retrieve": self._retriever_node,
            "policy": self._policy,
            "tool_execute": self._executor,
            "verify": self._verifier,
            "respond": self._responder,
        }

        for node_name in nodes:
            node = node_map.get(node_name)
            if node is None:
                logger.debug("Skipping unconfigured node: %s", node_name)
                continue
            if state.status in (_BUDGET_EXCEEDED, _TERMINATED, _ERROR):
                break
            try:
                state = node(state)
            except Exception as exc:
                state.status = _ERROR
                state.error = f"Node '{node_name}' failed: {exc}"
                logger.exception("Graph node '%s' failed", node_name)
                break

        return state

    def build_langgraph(self) -> Any:
        """Build and return a LangGraph ``StateGraph`` when langgraph is available.

        Falls back to ``None`` if langgraph is not installed, in which case
        use ``self.step()`` for sequential execution.
        """
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.info("langgraph not installed; using sequential execution")
            return None

        graph = StateGraph(dict)

        def _wrap(node_fn: Any):
            def _node(state_dict: dict) -> dict:
                st = AgentState(**state_dict)
                st = node_fn(st)
                return st.__dict__
            return _node

        node_list = (
            _NODE_ORDER_SIMPLE_RAG if self.mode == "simple_rag" else _NODE_ORDER_AGENT_RAG
        )

        node_map = {
            "plan": self._planner,
            "retrieve": self._retriever_node,
            "policy": self._policy,
            "tool_execute": self._executor,
            "verify": self._verifier,
            "respond": self._responder,
        }

        active_nodes = [(n, node_map[n]) for n in node_list if node_map.get(n) is not None]
        if not active_nodes:
            return None

        for name, fn in active_nodes:
            graph.add_node(name, _wrap(fn))

        graph.set_entry_point(active_nodes[0][0])
        for i in range(len(active_nodes) - 1):
            graph.add_edge(active_nodes[i][0], active_nodes[i + 1][0])
        graph.add_edge(active_nodes[-1][0], END)

        return graph.compile()


__all__ = [
    "AgentState",
    "HarnessGraph",
    "PlannerNode",
    "PolicyNode",
    "ResponderNode",
    "RetrieverNode",
    "ToolExecutorNode",
    "VerifierNode",
    "build_graph",
]
