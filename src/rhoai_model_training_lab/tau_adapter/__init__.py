"""τ-bench simulation adapter.

Bridges the official τ-bench banking simulation to the lab harness,
providing tool execution, episode management, schema validation,
and private-field filtering.  Imports τ2-bench through try/except
for graceful degradation when it is not installed.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version pinning and compatibility
# ---------------------------------------------------------------------------

SUPPORTED_TAU_VERSIONS = ("1.0.1", "1.0.0")
_TAU_MIN_PYTHON = (3, 11)

_tau_available = False
_tau_module: Any = None
_tau_version: str = ""

try:
    import tau2  # type: ignore[import-untyped]

    _tau_module = tau2
    _tau_version = getattr(tau2, "__version__", "unknown")
    _tau_available = True
    logger.info("τ2-bench loaded: version=%s", _tau_version)
except ImportError:
    logger.info(
        "τ2-bench not installed; TauSimulationAdapter will operate in "
        "stub mode.  Install with: pip install tau2-bench"
    )


def is_tau_available() -> bool:
    """Return True if the τ2-bench package is importable."""
    return _tau_available


def get_tau_version() -> str:
    """Return the installed τ2-bench version string, or '' if unavailable."""
    return _tau_version


def check_tau_compatibility() -> dict[str, Any]:
    """Check version compatibility with the lab harness.

    Returns a dict with ``compatible``, ``version``, ``warnings``, and
    ``errors`` keys.
    """
    result: dict[str, Any] = {
        "compatible": False,
        "version": _tau_version,
        "warnings": [],
        "errors": [],
    }

    if not _tau_available:
        result["errors"].append("τ2-bench is not installed")
        return result

    import sys

    if sys.version_info < _TAU_MIN_PYTHON:
        result["errors"].append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} "
            f"< minimum {_TAU_MIN_PYTHON[0]}.{_TAU_MIN_PYTHON[1]}"
        )

    if _tau_version not in SUPPORTED_TAU_VERSIONS and _tau_version != "unknown":
        result["warnings"].append(
            f"τ2-bench version {_tau_version} not in tested versions: "
            f"{SUPPORTED_TAU_VERSIONS}"
        )

    if not result["errors"]:
        result["compatible"] = True

    return result


# ---------------------------------------------------------------------------
# Private-field filtering
# ---------------------------------------------------------------------------

_PRIVATE_FIELD_PATTERNS = [
    re.compile(r"expected_action", re.IGNORECASE),
    re.compile(r"hidden_goal", re.IGNORECASE),
    re.compile(r"target_reward", re.IGNORECASE),
    re.compile(r"evaluator_state", re.IGNORECASE),
    re.compile(r"private_instructions?", re.IGNORECASE),
    re.compile(r"golden_doc", re.IGNORECASE),
    re.compile(r"reward_criteria", re.IGNORECASE),
    re.compile(r"success_condition", re.IGNORECASE),
    re.compile(r"grading_rubric", re.IGNORECASE),
    re.compile(r"expected_tool_calls?", re.IGNORECASE),
]


def filter_private_fields(task: dict[str, Any]) -> dict[str, Any]:
    """Remove private evaluator fields from a task dict.

    Ensures that hidden expected actions, golden document lists, reward
    criteria, and other evaluator-only fields never enter model payloads
    or training data.

    Parameters
    ----------
    task : dict
        A task or scenario dictionary that may contain private fields.

    Returns
    -------
    dict
        A copy of the task with private fields removed.
    """
    filtered = copy.deepcopy(task)
    _strip_private_recursive(filtered)
    return filtered


def _strip_private_recursive(obj: Any) -> None:
    """Recursively remove keys matching private-field patterns."""
    if isinstance(obj, dict):
        keys_to_remove = []
        for key in obj:
            if any(p.search(str(key)) for p in _PRIVATE_FIELD_PATTERNS):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del obj[key]
        for value in obj.values():
            _strip_private_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_private_recursive(item)


# ---------------------------------------------------------------------------
# Tool-call validation
# ---------------------------------------------------------------------------


def validate_tool_call(
    tool_name: str,
    args: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate a tool call against its schema.

    Parameters
    ----------
    tool_name : str
        Name of the tool being called.
    args : dict
        Arguments to the tool call.
    schema : dict
        The tool schema (OpenAI-style ``function`` definition).

    Returns
    -------
    dict
        ``{"valid": bool, "errors": list[str]}``.
    """
    errors: list[str] = []

    fn_def = schema.get("function", schema)
    schema_name = fn_def.get("name", "")
    if schema_name and schema_name != tool_name:
        errors.append(f"Tool name mismatch: expected '{schema_name}', got '{tool_name}'")

    params = fn_def.get("parameters", {})
    required = params.get("required", [])
    properties = params.get("properties", {})

    for req_param in required:
        if req_param not in args:
            errors.append(f"Missing required parameter: '{req_param}'")

    for arg_name, arg_value in args.items():
        if properties and arg_name not in properties:
            errors.append(f"Unknown parameter: '{arg_name}'")
            continue

        if arg_name in properties:
            prop_schema = properties[arg_name]
            expected_type = prop_schema.get("type", "")
            type_error = _check_type(arg_value, expected_type)
            if type_error:
                errors.append(f"Parameter '{arg_name}': {type_error}")

            if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
                errors.append(
                    f"Parameter '{arg_name}': value {arg_value!r} not in "
                    f"allowed values {prop_schema['enum']}"
                )

    return {"valid": len(errors) == 0, "errors": errors}


def _check_type(value: Any, expected_type: str) -> str | None:
    """Check a value against a JSON Schema type string."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return None
    if not isinstance(value, expected):
        return f"expected type '{expected_type}', got {type(value).__name__}"
    return None


# ---------------------------------------------------------------------------
# Simulation adapter
# ---------------------------------------------------------------------------


class TauSimulationAdapter:
    """Connect to the official τ banking simulation for tool execution.

    When τ2-bench is installed, this adapter delegates tool calls to the
    official simulation environment.  Otherwise it operates in stub mode,
    returning structured error messages.

    Parameters
    ----------
    domain : str
        τ-bench domain (default ``banking_knowledge``).
    tau_version : str
        Expected τ-bench version for compatibility checks.
    """

    def __init__(
        self,
        domain: str = "banking_knowledge",
        tau_version: str = "",
    ) -> None:
        self.domain = domain
        self.expected_version = tau_version
        self._env: Any = None
        self._tools: dict[str, dict[str, Any]] = {}
        self._episode_active: bool = False

        if _tau_available:
            compat = check_tau_compatibility()
            if not compat["compatible"]:
                logger.warning("τ-bench compatibility issues: %s", compat["errors"])
            for w in compat.get("warnings", []):
                logger.warning("τ-bench: %s", w)

    def connect(self, **kwargs: Any) -> None:
        """Initialise the simulation environment.

        Parameters
        ----------
        **kwargs
            Passed through to the τ environment constructor.
        """
        if not _tau_available:
            logger.warning("τ2-bench not available; running in stub mode")
            return

        try:
            env_cls = getattr(_tau_module, "BankingEnvironment", None)
            if env_cls is None:
                from tau2.domains import banking_knowledge  # type: ignore[import-untyped]
                env_cls = getattr(banking_knowledge, "Environment", None)

            if env_cls is not None:
                self._env = env_cls(domain=self.domain, **kwargs)
                logger.info("τ simulation environment connected: %s", self.domain)
            else:
                logger.warning(
                    "Could not find τ environment class; "
                    "inspect tau2 package structure"
                )
        except Exception:
            logger.exception("Failed to connect τ simulation environment")

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return tool schemas available in the current simulation.

        Preserves official tool-discovery behavior — tools are returned
        as the simulation provides them, not all upfront.
        """
        if self._env is not None:
            try:
                tools_fn = getattr(self._env, "get_tools", None)
                if tools_fn:
                    raw_tools = tools_fn()
                    tools = []
                    for t in raw_tools:
                        if isinstance(t, dict):
                            tools.append(t)
                        elif hasattr(t, "model_dump"):
                            tools.append(t.model_dump())
                        elif hasattr(t, "to_dict"):
                            tools.append(t.to_dict())
                        else:
                            tools.append({"name": str(t)})
                    self._tools = {
                        t.get("function", t).get("name", t.get("name", "")): t
                        for t in tools
                    }
                    return tools
            except Exception:
                logger.exception("Failed to discover tools from τ environment")

        return list(self._tools.values())

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call through the simulation.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute.
        args : dict
            Arguments for the tool call.

        Returns
        -------
        dict
            Tool execution result with ``status`` and ``result`` keys.

        Raises
        ------
        ValueError
            If the tool name is unknown or validation fails.
        RuntimeError
            If the simulation is not connected.
        """
        if tool_name in self._tools:
            schema = self._tools[tool_name]
            validation = validate_tool_call(tool_name, args, schema)
            if not validation["valid"]:
                return {
                    "status": "error",
                    "error": f"Validation failed: {'; '.join(validation['errors'])}",
                    "tool_name": tool_name,
                }

        if self._env is not None:
            try:
                exec_fn = getattr(self._env, "execute_tool", None)
                if exec_fn is None:
                    exec_fn = getattr(self._env, "step", None)

                if exec_fn:
                    result = exec_fn(tool_name, args)
                    if isinstance(result, dict):
                        return {"status": "ok", "result": result, "tool_name": tool_name}
                    elif hasattr(result, "model_dump"):
                        return {"status": "ok", "result": result.model_dump(), "tool_name": tool_name}
                    else:
                        return {"status": "ok", "result": str(result), "tool_name": tool_name}
            except Exception as exc:
                logger.exception("Tool execution failed: %s", tool_name)
                return {
                    "status": "error",
                    "error": str(exc),
                    "tool_name": tool_name,
                }

        logger.warning("Stub mode: tool '%s' not executed", tool_name)
        return {
            "status": "stub",
            "error": "τ simulation not connected; tool not executed",
            "tool_name": tool_name,
            "args": args,
        }

    def reset_episode(self) -> dict[str, Any]:
        """Reset the simulation environment for a new episode / task trial.

        Returns
        -------
        dict
            Reset status and initial observations.
        """
        if self._env is not None:
            try:
                reset_fn = getattr(self._env, "reset", None)
                if reset_fn:
                    obs = reset_fn()
                    self._episode_active = True
                    if isinstance(obs, dict):
                        return {"status": "ok", "observations": filter_private_fields(obs)}
                    return {"status": "ok", "observations": str(obs)}
            except Exception:
                logger.exception("Episode reset failed")
                return {"status": "error", "error": "Episode reset failed"}

        self._episode_active = True
        return {"status": "stub", "observations": {}}

    def end_episode(self) -> None:
        """Mark the current episode as ended."""
        self._episode_active = False


# ---------------------------------------------------------------------------
# Agent bridge for τ evaluation
# ---------------------------------------------------------------------------


class TauAgentBridge:
    """Bridge the official τ Agent / LLM interface to /v1/agent/step.

    The official τ evaluator calls an ``Agent`` or ``LLM`` interface to
    interact with the model under test.  This bridge translates those
    calls into HTTP requests to the lab's ``/v1/agent/step`` endpoint,
    allowing the same backend to serve both interactive use and
    automated evaluation.

    Parameters
    ----------
    backend_url : str
        Base URL of the lab backend (e.g. ``http://127.0.0.1:8000``).
    session_prefix : str
        Prefix for generated session IDs.
    mode : str
        Harness mode to request (``simple_rag`` or ``agent_rag``).
    use_examples : bool
        Whether to enable example retrieval.
    knowledge_access : str
        Knowledge access level (``no_knowledge``, ``rag``, ``full_kb``,
        ``golden_retrieval``).
    """

    def __init__(
        self,
        backend_url: str = "http://127.0.0.1:8000",
        session_prefix: str = "tau_eval",
        mode: str = "agent_rag",
        use_examples: bool = False,
        knowledge_access: str = "rag",
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.session_prefix = session_prefix
        self.mode = mode
        self.use_examples = use_examples
        self.knowledge_access = knowledge_access
        self._client: Any = None
        self._session_counter: int = 0

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=120.0)
        return self._client

    def new_session_id(self) -> str:
        """Generate a new unique session ID for an evaluation episode."""
        self._session_counter += 1
        return f"{self.session_prefix}_{self._session_counter}"

    def step(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        available_tools: list[dict[str, Any]] | None = None,
        tool_observations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send an agent step request to the backend.

        Parameters
        ----------
        session_id : str
            Session identifier.
        messages : list[dict]
            Message history.
        available_tools : list[dict] | None
            Currently available tool schemas.
        tool_observations : list[dict] | None
            Tool execution results from the simulation.

        Returns
        -------
        dict
            Parsed response from ``/v1/agent/step``.
        """
        client = self._ensure_client()

        for msg in messages:
            filter_private_fields(msg)

        payload: dict[str, Any] = {
            "session_id": session_id,
            "messages": messages,
            "mode": self.mode,
            "use_examples": self.use_examples,
            "knowledge_access": self.knowledge_access,
        }
        if available_tools:
            payload["available_tools"] = available_tools
        if tool_observations:
            payload["tool_observations"] = tool_observations

        url = f"{self.backend_url}/v1/agent/step"
        try:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.exception("Agent step request failed: %s", url)
            return {
                "status": "error",
                "error": str(exc),
                "session_id": session_id,
            }

    def reset(self, session_id: str | None = None) -> str:
        """Reset for a new evaluation episode.

        Returns the session_id (new or provided).
        """
        if session_id is None:
            session_id = self.new_session_id()
        return session_id

    def call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Implement the τ LLM interface.

        This method matches the signature expected by the official τ
        evaluator when it needs a raw LLM call.

        Parameters
        ----------
        messages : list[dict]
            Message history.
        tools : list[dict] | None
            Available tool schemas.

        Returns
        -------
        dict
            Response with ``content`` and optionally ``tool_calls``.
        """
        session_id = self.new_session_id()
        result = self.step(
            session_id=session_id,
            messages=messages,
            available_tools=tools,
        )

        message_data = result.get("message", {})
        if isinstance(message_data, dict):
            return {
                "content": message_data.get("content", ""),
                "tool_calls": result.get("tool_calls"),
            }
        return {"content": "", "tool_calls": result.get("tool_calls")}


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_adapter(config: dict[str, Any]) -> TauSimulationAdapter:
    """Create a TauSimulationAdapter from config.

    Parameters
    ----------
    config : dict
        Configuration dict, typically from ``eval.yaml`` or ``rag.yaml``.
    """
    tau_cfg = config.get("tau_episodes", config)
    adapter = TauSimulationAdapter(
        domain=tau_cfg.get("domain", "banking_knowledge"),
        tau_version=tau_cfg.get("tau_version", ""),
    )
    adapter.connect()
    return adapter


def create_bridge(config: dict[str, Any]) -> TauAgentBridge:
    """Create a TauAgentBridge from config.

    Parameters
    ----------
    config : dict
        Configuration dict with ``api`` section for backend URL.
    """
    api_cfg = config.get("api", {})
    host = api_cfg.get("host", "127.0.0.1")
    port = api_cfg.get("port", "8000")
    backend_url = f"http://{host}:{port}"

    harness_cfg = config.get("harness", {})
    return TauAgentBridge(
        backend_url=backend_url,
        mode=harness_cfg.get("mode", "agent_rag"),
        use_examples=harness_cfg.get("use_examples", False),
    )


__all__ = [
    "TauAgentBridge",
    "TauSimulationAdapter",
    "check_tau_compatibility",
    "create_adapter",
    "create_bridge",
    "filter_private_fields",
    "get_tau_version",
    "is_tau_available",
    "validate_tool_call",
]
