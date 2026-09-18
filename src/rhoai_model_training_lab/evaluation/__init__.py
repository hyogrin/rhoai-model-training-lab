"""Evaluation module for all three tracks.

Track A: prepared_diagnostics — holdout-based offline evaluation
Track B: tau_episodes — official τ-Knowledge end-to-end episodes
Track C: retention — ARC-Challenge capability preservation
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import httpx

from rhoai_model_training_lab.config import (
    PROJECT_ROOT,
    load_eval_config,
    load_yaml_config,
)
from rhoai_model_training_lab.schemas.evaluation import (
    ComparisonRow,
    EvalMetrics,
    EvalResult,
    EvalRunConfig,
    RetentionResult,
    TaskResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PreparedDiagnosticsEvaluator",
    "TauEpisodeRunner",
    "RetentionEvaluator",
    "ConfidenceCalculator",
    "EvalOrchestrator",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_config_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 over sorted JSON of a config dict."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    items: list[dict[str, Any]] = []
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"JSONL file not found: {fpath}")
    with open(fpath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d: %s", fpath, lineno, exc)
    return items


def _save_jsonl(items: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dicts as JSONL."""
    fpath = Path(path)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        for item in items:
            f.write(json.dumps(item, default=str) + "\n")
    return fpath


def _make_fingerprint(
    task_id: str,
    trial: int,
    model_hash: str,
    data_revision: str,
    config_hash: str,
) -> str:
    """Build a deterministic run fingerprint for deduplication and resume."""
    parts = f"{task_id}|{trial}|{model_hash}|{data_revision}|{config_hash}"
    return hashlib.sha256(parts.encode()).hexdigest()[:24]


def _resolve_endpoint_url(endpoint_name: str, eval_config: dict[str, Any]) -> str:
    """Resolve a logical endpoint name to a URL from endpoints config."""
    try:
        endpoints_cfg = load_yaml_config("configs/endpoints.example.yaml")
    except FileNotFoundError:
        try:
            endpoints_cfg = load_yaml_config("configs/endpoints.yaml")
        except FileNotFoundError:
            endpoints_cfg = {}

    endpoints = endpoints_cfg.get("endpoints", {})
    ep = endpoints.get(endpoint_name, {})
    url = ep.get("url", "")
    if not url:
        logger.warning("No URL configured for endpoint '%s'", endpoint_name)
    return url


def _call_model_endpoint(
    endpoint_url: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    timeout: float = 300.0,
    token: str = "",
    ca_bundle: str = "",
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat/completions endpoint.

    Returns the parsed JSON response body. Raises on HTTP or connection errors.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    verify: bool | str = True
    if ca_bundle:
        verify = ca_bundle

    url = endpoint_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    with httpx.Client(verify=verify, timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Track A: Prepared Diagnostics
# ---------------------------------------------------------------------------

class PreparedDiagnosticsEvaluator:
    """Evaluate models on holdout diagnostic samples with matched context.

    Results are labelled as **τ-Knowledge-derived lab diagnostics**, not
    official τ pass rates.  In ``matched_context`` mode every model variant
    receives the same KB excerpts and observations.
    """

    def __init__(self, eval_config: dict[str, Any]) -> None:
        self.eval_config = eval_config
        self.track_config = eval_config.get("tracks", {}).get("prepared_diagnostics", {})
        self.holdout_path = self.track_config.get(
            "holdout_path",
            "data/prepared/tau-knowledge-v1/canonical/validation.jsonl",
        )
        self.scoring_flags = self.track_config.get("scoring", {})
        self.judge_config = self.track_config.get("judge", {})
        self._holdout_samples: list[dict[str, Any]] | None = None

    # -- data loading -------------------------------------------------------

    def load_holdout(self) -> list[dict[str, Any]]:
        """Load holdout samples from the configured JSONL path."""
        holdout_abs = Path(self.holdout_path)
        if not holdout_abs.is_absolute():
            holdout_abs = PROJECT_ROOT / holdout_abs
        self._holdout_samples = _load_jsonl(holdout_abs)
        logger.info(
            "Loaded %d holdout samples from %s",
            len(self._holdout_samples),
            holdout_abs,
        )
        return self._holdout_samples

    @property
    def holdout_samples(self) -> list[dict[str, Any]]:
        if self._holdout_samples is None:
            self.load_holdout()
        assert self._holdout_samples is not None
        return self._holdout_samples

    # -- scoring ------------------------------------------------------------

    @staticmethod
    def score_answer_accuracy(predicted: str, expected: str) -> float:
        """Score answer accuracy using exact / normalised-substring match.

        Returns 1.0 for match, 0.0 otherwise.  A pluggable judge endpoint
        can override this with rubric-based grading.
        """
        if not expected:
            return 1.0 if not predicted else 0.0
        pred_norm = predicted.strip().lower()
        exp_norm = expected.strip().lower()
        if pred_norm == exp_norm:
            return 1.0
        if exp_norm in pred_norm:
            return 0.8
        return 0.0

    @staticmethod
    def score_condition_accuracy(predicted_conditions: list[str], expected_conditions: list[str]) -> float:
        """Score condition coverage (Jaccard on normalised condition strings)."""
        if not expected_conditions:
            return 1.0
        pred_set = {c.strip().lower() for c in predicted_conditions}
        exp_set = {c.strip().lower() for c in expected_conditions}
        if not exp_set:
            return 1.0
        intersection = pred_set & exp_set
        union = pred_set | exp_set
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def score_tool_name(predicted: str, expected: str) -> float:
        """Binary match on predicted vs expected tool name."""
        return 1.0 if predicted.strip() == expected.strip() else 0.0

    @staticmethod
    def score_tool_args(predicted_args: dict[str, Any], expected_args: dict[str, Any]) -> float:
        """Key-level argument match score."""
        if not expected_args:
            return 1.0
        all_keys = set(expected_args.keys())
        matched = sum(
            1
            for k in all_keys
            if str(predicted_args.get(k, "")).strip() == str(expected_args[k]).strip()
        )
        return matched / len(all_keys)

    @staticmethod
    def score_grounding(citations: list[str], expected_doc_ids: list[str]) -> float:
        """Score grounding by recall of expected document references."""
        if not expected_doc_ids:
            return 1.0
        cited_set = {c.strip() for c in citations}
        expected_set = {d.strip() for d in expected_doc_ids}
        recall = len(cited_set & expected_set) / len(expected_set)
        return recall

    # -- evaluation pipeline ------------------------------------------------

    def evaluate_sample(
        self,
        sample: dict[str, Any],
        model_response: dict[str, Any],
    ) -> dict[str, float]:
        """Score a single sample against a model response.

        Args:
            sample: The holdout sample with expected fields.
            model_response: Parsed model response (answer, tool_calls, citations).

        Returns:
            Dictionary of metric_name → score.
        """
        scores: dict[str, float] = {}

        if self.scoring_flags.get("answer_accuracy", True):
            scores["answer_accuracy"] = self.score_answer_accuracy(
                model_response.get("answer", ""),
                sample.get("expected_answer", ""),
            )

        if self.scoring_flags.get("condition_check", True):
            scores["condition_accuracy"] = self.score_condition_accuracy(
                model_response.get("conditions", []),
                sample.get("expected_conditions", []),
            )

        if self.scoring_flags.get("tool_name_match", True):
            pred_tool = ""
            tool_calls = model_response.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]
                pred_tool = tc.get("function", {}).get("name", tc.get("name", ""))
            scores["tool_name_accuracy"] = self.score_tool_name(
                pred_tool,
                sample.get("expected_tool", ""),
            )

        if self.scoring_flags.get("tool_args_match", True):
            pred_args: dict[str, Any] = {}
            tool_calls = model_response.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0]
                fn = tc.get("function", {})
                raw = fn.get("arguments", "{}")
                if isinstance(raw, str):
                    try:
                        pred_args = json.loads(raw)
                    except json.JSONDecodeError:
                        pred_args = {}
                elif isinstance(raw, dict):
                    pred_args = raw
            scores["tool_args_accuracy"] = self.score_tool_args(
                pred_args,
                sample.get("expected_tool_args", {}),
            )

        if self.scoring_flags.get("grounding_check", True):
            scores["grounding_score"] = self.score_grounding(
                model_response.get("citations", []),
                sample.get("expected_doc_ids", []),
            )

        return scores

    def run(
        self,
        model_endpoint: str,
        *,
        variant: str = "base",
        limit: int | None = None,
        config: EvalRunConfig | None = None,
    ) -> EvalResult:
        """Run prepared diagnostics evaluation against a model endpoint.

        In matched_context mode, KB excerpts from the holdout sample are
        injected directly so that Base, LoRA, and OSFT receive identical
        context.

        Args:
            model_endpoint: URL of the model serving endpoint.
            variant: Model variant label (base, lora, osft).
            limit: Max samples to evaluate (None = all).
            config: Optional run config for fingerprinting.

        Returns:
            EvalResult with task_results and aggregated metrics.
        """
        samples = self.holdout_samples
        if limit is not None:
            samples = samples[:limit]

        config_hash = _compute_config_hash(self.track_config)
        run_id = str(uuid.uuid4())
        task_results: list[TaskResult] = []
        score_accumulators: dict[str, list[float]] = defaultdict(list)

        matched_context = self.track_config.get("matched_context", True)

        for idx, sample in enumerate(samples):
            task_id = sample.get("sample_id", f"diag-{idx}")
            t0 = time.monotonic()

            try:
                messages = list(sample.get("messages", []))
                if matched_context and sample.get("context_documents"):
                    ctx_text = "\n\n".join(sample["context_documents"])
                    messages.insert(0, {
                        "role": "system",
                        "content": f"Reference documents:\n{ctx_text}",
                    })

                tools = sample.get("tools")
                raw_resp = _call_model_endpoint(
                    model_endpoint,
                    messages,
                    tools=tools,
                    max_tokens=config.max_tokens if config else 8192,
                )

                choice = raw_resp.get("choices", [{}])[0]
                msg = choice.get("message", {})
                model_response = {
                    "answer": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                    "conditions": [],
                    "citations": [],
                }

                scores = self.evaluate_sample(sample, model_response)
                wall = time.monotonic() - t0

                usage = raw_resp.get("usage", {})
                task_results.append(TaskResult(
                    task_id=task_id,
                    trial=0,
                    success=all(v >= 0.5 for v in scores.values()),
                    reward=statistics.mean(scores.values()) if scores else 0.0,
                    turns=1,
                    tool_calls=len(msg.get("tool_calls", [])),
                    total_tokens=usage.get("total_tokens", 0),
                    wall_time_seconds=wall,
                    model_latency_ms=wall * 1000,
                ))
                for k, v in scores.items():
                    score_accumulators[k].append(v)

            except Exception as exc:
                wall = time.monotonic() - t0
                error_type = "api_error"
                if isinstance(exc, httpx.TimeoutException):
                    error_type = "timeout"
                logger.error("Diagnostic evaluation failed for %s: %s", task_id, exc)
                task_results.append(TaskResult(
                    task_id=task_id,
                    trial=0,
                    success=False,
                    reward=0.0,
                    wall_time_seconds=wall,
                    error=str(exc),
                    error_type=error_type,
                ))

        metrics = self._aggregate_metrics(task_results, score_accumulators, variant)

        return EvalResult(
            run_id=run_id,
            track="prepared_diagnostics",
            variant=variant,
            config=config,
            metrics=metrics,
            task_results=task_results,
            data_run_id=config.bundle_id if config else "",
            model_hash=config.model_hash if config else "",
            execution_status="completed",
        )

    @staticmethod
    def _aggregate_metrics(
        task_results: list[TaskResult],
        score_accumulators: dict[str, list[float]],
        variant: str,
    ) -> EvalMetrics:
        """Aggregate per-sample scores into EvalMetrics."""
        total = len(task_results)
        succeeded = sum(1 for t in task_results if t.success)
        failed = sum(1 for t in task_results if t.error is not None)
        timed_out = sum(1 for t in task_results if t.error_type == "timeout")

        error_counts: dict[str, int] = defaultdict(int)
        for t in task_results:
            if t.error_type:
                error_counts[t.error_type] += 1

        metrics = EvalMetrics(
            track="prepared_diagnostics",
            variant=variant,
            total_tasks=total,
            attempted=total,
            completed=total - failed,
            succeeded=succeeded,
            failed=failed,
            timed_out=timed_out,
            task_success_rate=succeeded / total if total else None,
            answer_accuracy=(
                statistics.mean(score_accumulators["answer_accuracy"])
                if score_accumulators.get("answer_accuracy")
                else None
            ),
            tool_name_accuracy=(
                statistics.mean(score_accumulators["tool_name_accuracy"])
                if score_accumulators.get("tool_name_accuracy")
                else None
            ),
            tool_args_accuracy=(
                statistics.mean(score_accumulators["tool_args_accuracy"])
                if score_accumulators.get("tool_args_accuracy")
                else None
            ),
            grounding_score=(
                statistics.mean(score_accumulators["grounding_score"])
                if score_accumulators.get("grounding_score")
                else None
            ),
            condition_accuracy=(
                statistics.mean(score_accumulators["condition_accuracy"])
                if score_accumulators.get("condition_accuracy")
                else None
            ),
            mean_turns=statistics.mean(t.turns for t in task_results) if task_results else 0.0,
            mean_tool_calls=statistics.mean(t.tool_calls for t in task_results) if task_results else 0.0,
            mean_tokens=statistics.mean(t.total_tokens for t in task_results) if task_results else 0.0,
            mean_wall_time_seconds=(
                statistics.mean(t.wall_time_seconds for t in task_results) if task_results else 0.0
            ),
            mean_model_latency_ms=(
                statistics.mean(t.model_latency_ms for t in task_results) if task_results else 0.0
            ),
            error_counts=dict(error_counts),
        )
        return metrics


# ---------------------------------------------------------------------------
# Track B: Official τ Episodes
# ---------------------------------------------------------------------------

class TauEpisodeRunner:
    """Run official τ-Knowledge episodes against live model endpoints.

    Supports all variant combinations: base/lora/osft × no_knowledge/rag.
    Uses official runner/grader semantics with episode resets, session
    isolation, and pass^k (NOT pass@k) computation.  Failures and timeouts
    are always kept in the denominator.
    """

    def __init__(self, eval_config: dict[str, Any]) -> None:
        self.eval_config = eval_config
        self.track_config = eval_config.get("tracks", {}).get("tau_episodes", {})
        self.budgets = self.track_config.get("budgets", {})
        self.grading_config = self.track_config.get("grading", {})
        self.confidence_config = self.track_config.get("confidence", {})

    # -- pass^k computation -------------------------------------------------

    @staticmethod
    def compute_pass_k(task_trials: dict[str, list[bool]], k: int) -> float:
        r"""Compute pass^k: consistent success across k trials.

        pass^k = Σ_task [ ∏_{i=1}^{k} success_i ] / |tasks|

        Unlike code-generation pass@k (best-of-k), pass^k requires the model
        to succeed on *every* trial for a given task.  A task with any failure
        trial scores 0 for that task.

        Args:
            task_trials: Mapping of task_id → list of boolean success outcomes.
            k: Number of trials.

        Returns:
            pass^k score in [0, 1].
        """
        if not task_trials:
            return 0.0
        task_scores: list[float] = []
        for _task_id, trials in task_trials.items():
            effective = trials[:k]
            if len(effective) < k:
                task_scores.append(0.0)
            else:
                task_scores.append(1.0 if all(effective) else 0.0)
        return statistics.mean(task_scores) if task_scores else 0.0

    # -- episode execution --------------------------------------------------

    def _run_single_episode(
        self,
        task_id: str,
        trial: int,
        variant_config: dict[str, Any],
        run_config: EvalRunConfig,
    ) -> TaskResult:
        """Execute a single τ episode for one task and trial.

        Each episode:
        1. Resets the simulation environment to the task's initial state.
        2. Runs an agent loop: model inference → tool execution → observation.
        3. Grades the episode using the official grader.
        4. Records the result (failures/timeouts stay in denominator).

        Args:
            task_id: Identifier of the τ task.
            trial: Trial number (0-indexed).
            variant_config: Variant-specific configuration (endpoint, mode, etc.).
            run_config: The overall run configuration.

        Returns:
            TaskResult for this (task_id, trial).
        """
        endpoint_name = variant_config.get("model_endpoint", "base")
        endpoint_url = run_config.model_endpoint or _resolve_endpoint_url(
            endpoint_name, self.eval_config,
        )
        knowledge_access = variant_config.get("knowledge_access", run_config.knowledge_access)
        mode = variant_config.get("mode", run_config.mode)
        use_examples = variant_config.get("use_examples", run_config.use_examples)

        max_turns = self.budgets.get("max_turns", run_config.max_turns)
        max_tool_calls = self.budgets.get("max_tool_calls", run_config.max_tool_calls)
        max_tokens = self.budgets.get("max_tokens", run_config.max_tokens)
        max_wall_time = self.budgets.get("max_wall_time_seconds", run_config.max_wall_time_seconds)

        session_id = f"{task_id}-trial{trial}-{uuid.uuid4().hex[:8]}"
        t0 = time.monotonic()
        turns = 0
        tool_call_count = 0
        total_tokens = 0
        actions: list[str] = []
        error: str | None = None
        error_type: str | None = None
        reward = 0.0
        success = False

        try:
            env_state = self._reset_environment(task_id, session_id)

            messages: list[dict[str, Any]] = []
            if env_state.get("system_message"):
                messages.append({"role": "system", "content": env_state["system_message"]})
            if env_state.get("user_message"):
                messages.append({"role": "user", "content": env_state["user_message"]})

            available_tools = env_state.get("tools", [])

            episode_done = False
            while not episode_done and turns < max_turns:
                wall_elapsed = time.monotonic() - t0
                if wall_elapsed > max_wall_time:
                    error = f"Wall-time budget exceeded: {wall_elapsed:.1f}s > {max_wall_time}s"
                    error_type = "timeout"
                    break

                if tool_call_count >= max_tool_calls:
                    error = f"Tool-call budget exceeded: {tool_call_count} >= {max_tool_calls}"
                    error_type = "budget_exceeded"
                    break

                model_t0 = time.monotonic()
                try:
                    resp = _call_model_endpoint(
                        endpoint_url,
                        messages,
                        tools=available_tools if knowledge_access != "no_knowledge" or available_tools else None,
                        max_tokens=max_tokens,
                        timeout=float(max_wall_time - wall_elapsed),
                    )
                except httpx.TimeoutException:
                    error = "Model endpoint timeout"
                    error_type = "timeout"
                    break
                except httpx.HTTPStatusError as exc:
                    error = f"Model endpoint HTTP {exc.response.status_code}"
                    error_type = "api_error"
                    break

                model_latency = (time.monotonic() - model_t0) * 1000
                usage = resp.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)
                turns += 1

                choice = resp.get("choices", [{}])[0]
                finish_reason = choice.get("finish_reason", "")
                assistant_msg = choice.get("message", {})
                messages.append(assistant_msg)

                resp_tool_calls = assistant_msg.get("tool_calls", [])
                if resp_tool_calls:
                    for tc in resp_tool_calls:
                        tool_call_count += 1
                        fn = tc.get("function", {})
                        tool_name = fn.get("name", "")
                        tool_args_raw = fn.get("arguments", "{}")
                        actions.append(tool_name)

                        try:
                            tool_args = (
                                json.loads(tool_args_raw)
                                if isinstance(tool_args_raw, str)
                                else tool_args_raw
                            )
                        except json.JSONDecodeError:
                            tool_args = {}

                        tool_result = self._execute_tool(
                            session_id, tool_name, tool_args,
                        )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": tool_name,
                            "content": json.dumps(tool_result, default=str),
                        })

                        if tool_result.get("__episode_done"):
                            episode_done = True
                            break
                else:
                    user_response = self._get_user_response(
                        session_id, assistant_msg.get("content", ""),
                    )
                    if user_response.get("__episode_done"):
                        episode_done = True
                    else:
                        messages.append({
                            "role": "user",
                            "content": user_response.get("content", ""),
                        })

            grade = self._grade_episode(task_id, session_id, messages, actions)
            reward = grade.get("reward", 0.0)
            success = grade.get("success", False)

        except Exception as exc:
            logger.error("Episode %s trial %d failed: %s", task_id, trial, exc)
            error = str(exc)
            error_type = "api_error"

        wall_time = time.monotonic() - t0

        return TaskResult(
            task_id=task_id,
            trial=trial,
            success=success,
            reward=reward,
            actions=actions,
            turns=turns,
            tool_calls=tool_call_count,
            total_tokens=total_tokens,
            wall_time_seconds=wall_time,
            error=error,
            error_type=error_type,
        )

    # -- environment interaction (adapter hooks) ----------------------------

    def _reset_environment(self, task_id: str, session_id: str) -> dict[str, Any]:
        """Reset the τ simulation environment for a new episode.

        This is an adapter point.  A concrete τ-bench integration overrides
        this to call the official simulator's reset API.

        Returns:
            Dict with keys: system_message, user_message, tools, initial_state.
        """
        logger.info("Resetting environment for task=%s session=%s", task_id, session_id)
        return {
            "system_message": "",
            "user_message": "",
            "tools": [],
            "initial_state": {},
        }

    def _execute_tool(
        self,
        session_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool call in the simulation environment.

        Adapter point for official τ simulation tool execution.

        Returns:
            Tool result dict.  May include ``__episode_done: True`` to signal
            that the episode should terminate after this tool.
        """
        logger.debug("Tool call in session %s: %s(%s)", session_id, tool_name, tool_args)
        return {"status": "not_implemented", "tool": tool_name}

    def _get_user_response(
        self,
        session_id: str,
        assistant_text: str,
    ) -> dict[str, Any]:
        """Get the next user-simulator response.

        Adapter point for the official τ user simulator.

        Returns:
            Dict with ``content`` and optionally ``__episode_done: True``.
        """
        logger.debug("User response request in session %s", session_id)
        return {"content": "", "__episode_done": True}

    def _grade_episode(
        self,
        task_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        actions: list[str],
    ) -> dict[str, Any]:
        """Grade a completed episode using official τ grader semantics.

        Adapter point for the official τ grader/scorer.

        Returns:
            Dict with ``success: bool`` and ``reward: float``.
        """
        logger.debug("Grading episode task=%s session=%s", task_id, session_id)
        return {"success": False, "reward": 0.0}

    # -- full variant run ---------------------------------------------------

    def run_variant(
        self,
        variant_config: dict[str, Any],
        run_config: EvalRunConfig,
        *,
        limit: int | None = None,
        trials: int | None = None,
        resume_state: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Run all episodes for a single variant.

        Args:
            variant_config: Variant definition from eval.yaml.
            run_config: Run-level configuration.
            limit: Override max tasks (None = all).
            trials: Override trial count.
            resume_state: Previously saved partial state to resume from.

        Returns:
            Completed EvalResult.
        """
        variant_name = variant_config.get("name", "unknown")
        task_ids = run_config.task_ids or self._discover_tasks()
        effective_limit = limit or run_config.limit
        if effective_limit is not None:
            task_ids = task_ids[:effective_limit]
        effective_trials = trials or run_config.trials

        config_hash = _compute_config_hash({
            "variant": variant_name,
            "budgets": self.budgets,
            "grading": self.grading_config,
            "trials": effective_trials,
        })

        completed_fingerprints: set[str] = set()
        existing_results: list[TaskResult] = []
        if resume_state:
            for tr in resume_state.get("task_results", []):
                fp = _make_fingerprint(
                    tr["task_id"], tr["trial"],
                    run_config.model_hash, run_config.data_revision, config_hash,
                )
                completed_fingerprints.add(fp)
                existing_results.append(TaskResult(**tr))
            logger.info(
                "Resuming %s: %d completed results loaded",
                variant_name,
                len(existing_results),
            )

        run_id = resume_state.get("run_id", str(uuid.uuid4())) if resume_state else str(uuid.uuid4())
        task_results = list(existing_results)
        task_trial_map: dict[str, list[bool]] = defaultdict(list)

        for tr in existing_results:
            task_trial_map[tr.task_id].append(tr.success)

        for task_id in task_ids:
            for trial in range(effective_trials):
                fp = _make_fingerprint(
                    task_id, trial,
                    run_config.model_hash, run_config.data_revision, config_hash,
                )
                if fp in completed_fingerprints:
                    continue

                logger.info(
                    "Running episode: variant=%s task=%s trial=%d/%d",
                    variant_name, task_id, trial + 1, effective_trials,
                )
                result = self._run_single_episode(task_id, trial, variant_config, run_config)
                task_results.append(result)
                task_trial_map[task_id].append(result.success)

        pass_k = self.compute_pass_k(task_trial_map, effective_trials)

        metrics = self._aggregate_episode_metrics(
            task_results, variant_name, pass_k, effective_trials,
        )

        return EvalResult(
            run_id=run_id,
            track="tau_episodes",
            variant=variant_name,
            config=run_config,
            metrics=metrics,
            task_results=task_results,
            data_run_id=run_config.bundle_id,
            model_hash=run_config.model_hash,
            grader_version=self.grading_config.get("grader_version", ""),
            scorer_revision=self.grading_config.get("scorer_revision", ""),
            execution_status="completed",
        )

    def _discover_tasks(self) -> list[str]:
        """Discover available τ task IDs from the pinned release.

        Override this with actual τ-bench task discovery.
        """
        logger.warning(
            "Task discovery not connected to τ-bench. "
            "Provide task_ids in run config or override _discover_tasks()."
        )
        return []

    @staticmethod
    def _aggregate_episode_metrics(
        task_results: list[TaskResult],
        variant: str,
        pass_k: float,
        k: int,
    ) -> EvalMetrics:
        """Aggregate episode-level task results."""
        total = len(task_results)
        succeeded = sum(1 for t in task_results if t.success)
        failed = sum(1 for t in task_results if t.error is not None)
        timed_out = sum(1 for t in task_results if t.error_type == "timeout")
        budget_exceeded = sum(1 for t in task_results if t.error_type == "budget_exceeded")

        error_counts: dict[str, int] = defaultdict(int)
        for t in task_results:
            if t.error_type:
                error_counts[t.error_type] += 1

        unique_tasks = {t.task_id for t in task_results}

        return EvalMetrics(
            track="tau_episodes",
            variant=variant,
            total_tasks=len(unique_tasks),
            attempted=total,
            completed=total - failed,
            succeeded=succeeded,
            failed=failed,
            timed_out=timed_out,
            budget_exceeded=budget_exceeded,
            task_success_rate=succeeded / total if total else None,
            pass_k=pass_k,
            pass_k_k=k,
            mean_turns=statistics.mean(t.turns for t in task_results) if task_results else 0.0,
            mean_tool_calls=statistics.mean(t.tool_calls for t in task_results) if task_results else 0.0,
            mean_tokens=statistics.mean(t.total_tokens for t in task_results) if task_results else 0.0,
            mean_wall_time_seconds=(
                statistics.mean(t.wall_time_seconds for t in task_results) if task_results else 0.0
            ),
            error_counts=dict(error_counts),
        )


# ---------------------------------------------------------------------------
# Track C: Retention (ARC-Challenge)
# ---------------------------------------------------------------------------

class RetentionEvaluator:
    """Evaluate capability retention on ARC-Challenge.

    KB and RAG are disabled.  Computes
    ``retention_delta_pp = 100 * (adapted_accuracy - base_accuracy)``.
    """

    def __init__(self, eval_config: dict[str, Any]) -> None:
        self.eval_config = eval_config
        self.track_config = eval_config.get("tracks", {}).get("retention", {})
        self.benchmarks = self.track_config.get("benchmarks", [])

    def load_arc_challenge(
        self,
        *,
        sample_size: int = 100,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Load ARC-Challenge samples for retention evaluation.

        This attempts to load via the ``datasets`` library from HuggingFace.
        Falls back to a local cache if unavailable.

        Args:
            sample_size: Number of samples to use.
            seed: Random seed for reproducible sampling.

        Returns:
            List of ARC-Challenge question dicts.
        """
        try:
            import datasets as hf_datasets

            ds = hf_datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            ds_shuffled = ds.shuffle(seed=seed)
            samples = [dict(row) for row in ds_shuffled.select(range(min(sample_size, len(ds_shuffled))))]
            logger.info("Loaded %d ARC-Challenge samples from HuggingFace", len(samples))
            return samples
        except Exception as exc:
            logger.warning("Could not load ARC-Challenge from HuggingFace: %s", exc)

        cache_path = PROJECT_ROOT / "data" / "benchmarks" / "arc_challenge.jsonl"
        if cache_path.exists():
            all_items = _load_jsonl(cache_path)
            import random

            rng = random.Random(seed)
            rng.shuffle(all_items)
            samples = all_items[:sample_size]
            logger.info("Loaded %d ARC-Challenge samples from local cache", len(samples))
            return samples

        raise FileNotFoundError(
            "ARC-Challenge dataset unavailable. Install `datasets` or provide "
            f"a local cache at {cache_path}"
        )

    def evaluate_variant(
        self,
        model_endpoint: str,
        variant: str,
        samples: list[dict[str, Any]],
        *,
        base_accuracy: float | None = None,
    ) -> RetentionResult:
        """Evaluate a single model variant on ARC-Challenge.

        Sends each question as a zero-shot generated-answer prompt.  The model
        must respond with the correct answer choice letter (A, B, C, D).
        KB and banking system prompts are disabled as per spec.

        Args:
            model_endpoint: URL of the model serving endpoint.
            variant: Model variant label (base, lora, osft).
            samples: List of ARC-Challenge question dicts.
            base_accuracy: Pre-computed base accuracy for delta calculation.

        Returns:
            RetentionResult with accuracy and delta.
        """
        correct = 0
        total = len(samples)

        for sample in samples:
            question = sample.get("question", "")
            choices = sample.get("choices", {})
            answer_key = sample.get("answerKey", "")

            choice_labels = choices.get("label", [])
            choice_texts = choices.get("text", [])
            formatted_choices = "\n".join(
                f"{label}. {text}"
                for label, text in zip(choice_labels, choice_texts)
            )

            prompt_text = (
                f"Answer the following question by responding with only the letter "
                f"of the correct answer (A, B, C, or D).\n\n"
                f"Question: {question}\n{formatted_choices}\n\nAnswer:"
            )

            messages = [{"role": "user", "content": prompt_text}]

            try:
                resp = _call_model_endpoint(
                    model_endpoint,
                    messages,
                    max_tokens=16,
                    temperature=0.0,
                    timeout=60.0,
                )
                choice = resp.get("choices", [{}])[0]
                answer_text = choice.get("message", {}).get("content", "").strip().upper()
                predicted = answer_text[:1] if answer_text else ""
                if predicted == answer_key.upper():
                    correct += 1
            except Exception as exc:
                logger.warning("ARC-Challenge inference error for %s: %s", variant, exc)

        accuracy = correct / total if total else 0.0
        delta = 100.0 * (accuracy - base_accuracy) if base_accuracy is not None else 0.0

        return RetentionResult(
            benchmark="arc_challenge",
            variant=variant,
            sample_size=total,
            accuracy=accuracy,
            base_accuracy=base_accuracy or 0.0,
            retention_delta_pp=delta,
        )

    def run(
        self,
        endpoint_map: dict[str, str],
        *,
        sample_size: int | None = None,
        seed: int | None = None,
    ) -> EvalResult:
        """Run retention evaluation for all variants.

        Args:
            endpoint_map: Mapping of variant name → endpoint URL
                          (e.g. ``{"base": "...", "lora": "...", "osft": "..."}``).
            sample_size: Override sample size.
            seed: Override random seed.

        Returns:
            EvalResult with retention_results.
        """
        bench_cfg = next(
            (b for b in self.benchmarks if b.get("name") == "arc_challenge"),
            {},
        )
        effective_size = sample_size or bench_cfg.get("sample_size", 100)
        effective_seed = seed or bench_cfg.get("seed", 42)

        samples = self.load_arc_challenge(sample_size=effective_size, seed=effective_seed)
        retention_results: list[RetentionResult] = []

        base_url = endpoint_map.get("base", "")
        base_result: RetentionResult | None = None
        if base_url:
            base_result = self.evaluate_variant(base_url, "base", samples)
            retention_results.append(base_result)
            logger.info("Base retention accuracy: %.4f", base_result.accuracy)

        for variant_name in ("lora", "osft"):
            url = endpoint_map.get(variant_name)
            if not url:
                continue
            result = self.evaluate_variant(
                url,
                variant_name,
                samples,
                base_accuracy=base_result.accuracy if base_result else None,
            )
            retention_results.append(result)
            logger.info(
                "%s retention accuracy: %.4f (delta: %+.2f pp)",
                variant_name,
                result.accuracy,
                result.retention_delta_pp,
            )

        return EvalResult(
            run_id=str(uuid.uuid4()),
            track="retention",
            variant="all",
            retention_results=retention_results,
            execution_status="completed",
        )


# ---------------------------------------------------------------------------
# Confidence Calculator
# ---------------------------------------------------------------------------

class ConfidenceCalculator:
    """Paired bootstrap confidence intervals for task-level results.

    Preserves the dependence structure between repeated trials of the same
    task by resampling *tasks* (not individual trials).
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed

    def compute_ci(
        self,
        task_results: list[TaskResult],
    ) -> tuple[float, float, float]:
        """Compute bootstrap CI for the task success rate.

        Args:
            task_results: All task results (may have multiple trials per task).

        Returns:
            Tuple of (point_estimate, ci_lower, ci_upper).
        """
        import random

        task_successes: dict[str, list[bool]] = defaultdict(list)
        for tr in task_results:
            task_successes[tr.task_id].append(tr.success)

        task_ids = list(task_successes.keys())
        n_tasks = len(task_ids)
        if n_tasks == 0:
            return 0.0, 0.0, 0.0

        def _task_rate(tid: str) -> float:
            trials = task_successes[tid]
            return sum(trials) / len(trials)

        point_estimate = statistics.mean(_task_rate(tid) for tid in task_ids)

        rng = random.Random(self.seed)
        bootstrap_means: list[float] = []
        for _ in range(self.n_bootstrap):
            sampled = rng.choices(task_ids, k=n_tasks)
            mean_val = statistics.mean(_task_rate(tid) for tid in sampled)
            bootstrap_means.append(mean_val)

        bootstrap_means.sort()
        lower_idx = max(0, int(math.floor(self.alpha / 2 * self.n_bootstrap)))
        upper_idx = min(
            self.n_bootstrap - 1,
            int(math.ceil((1 - self.alpha / 2) * self.n_bootstrap)) - 1,
        )
        ci_lower = bootstrap_means[lower_idx]
        ci_upper = bootstrap_means[upper_idx]

        return point_estimate, ci_lower, ci_upper

    def compute_paired_ci(
        self,
        results_a: list[TaskResult],
        results_b: list[TaskResult],
    ) -> tuple[float, float, float]:
        """Compute paired bootstrap CI for the difference in success rates.

        Resamples tasks (preserving dependence between variants) and computes
        the distribution of (rate_A - rate_B).

        Args:
            results_a: Task results for variant A.
            results_b: Task results for variant B.

        Returns:
            Tuple of (point_estimate_diff, ci_lower, ci_upper).
        """
        import random

        rates_a: dict[str, float] = {}
        for tr in results_a:
            rates_a.setdefault(tr.task_id, []).append(tr.success)  # type: ignore[assignment]
        for tid in list(rates_a):
            trials = rates_a[tid]  # type: ignore[arg-type]
            rates_a[tid] = sum(trials) / len(trials)  # type: ignore[arg-type]

        rates_b: dict[str, float] = {}
        for tr in results_b:
            rates_b.setdefault(tr.task_id, []).append(tr.success)  # type: ignore[assignment]
        for tid in list(rates_b):
            trials = rates_b[tid]  # type: ignore[arg-type]
            rates_b[tid] = sum(trials) / len(trials)  # type: ignore[arg-type]

        common_tasks = sorted(set(rates_a) & set(rates_b))
        n_tasks = len(common_tasks)
        if n_tasks == 0:
            return 0.0, 0.0, 0.0

        point_diff = statistics.mean(
            rates_a[tid] - rates_b[tid] for tid in common_tasks
        )

        rng = random.Random(self.seed)
        diffs: list[float] = []
        for _ in range(self.n_bootstrap):
            sampled = rng.choices(common_tasks, k=n_tasks)
            diff = statistics.mean(rates_a[tid] - rates_b[tid] for tid in sampled)
            diffs.append(diff)

        diffs.sort()
        lower_idx = max(0, int(math.floor(self.alpha / 2 * self.n_bootstrap)))
        upper_idx = min(
            self.n_bootstrap - 1,
            int(math.ceil((1 - self.alpha / 2) * self.n_bootstrap)) - 1,
        )

        return point_diff, diffs[lower_idx], diffs[upper_idx]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EvalOrchestrator:
    """Orchestrate evaluation across all three tracks.

    Provides ``run_track`` for individual tracks and ``run_all_tracks`` for
    a complete evaluation sweep.  Handles ``--limit`` and ``--trials`` flags,
    result persistence, and optional confidence interval computation.
    """

    TRACK_NAMES = ("prepared_diagnostics", "tau_episodes", "retention")

    def __init__(self, eval_config: dict[str, Any] | None = None) -> None:
        if eval_config is None:
            eval_config = load_eval_config()
        self.eval_config = eval_config

        self.diagnostics_evaluator = PreparedDiagnosticsEvaluator(eval_config)
        self.episode_runner = TauEpisodeRunner(eval_config)
        self.retention_evaluator = RetentionEvaluator(eval_config)
        self.confidence_calculator = ConfidenceCalculator(
            n_bootstrap=eval_config.get("tracks", {})
            .get("tau_episodes", {})
            .get("confidence", {})
            .get("n_bootstrap", 10_000),
            alpha=eval_config.get("tracks", {})
            .get("tau_episodes", {})
            .get("confidence", {})
            .get("alpha", 0.05),
        )

        self._output_dir = Path(
            eval_config.get("general", {}).get("output_dir", "data/evaluation/results")
        )
        if not self._output_dir.is_absolute():
            self._output_dir = PROJECT_ROOT / self._output_dir

    def run_track(
        self,
        track_name: str,
        config: EvalRunConfig,
        *,
        limit: int | None = None,
        trials: int | None = None,
        endpoint_map: dict[str, str] | None = None,
    ) -> list[EvalResult]:
        """Run a single evaluation track.

        Args:
            track_name: One of 'prepared_diagnostics', 'tau_episodes', 'retention'.
            config: Evaluation run configuration.
            limit: Override number of tasks.
            trials: Override number of trials.
            endpoint_map: Mapping of variant → endpoint URL.

        Returns:
            List of EvalResult (one per variant for tau_episodes, one otherwise).
        """
        if track_name not in self.TRACK_NAMES:
            raise ValueError(
                f"Unknown track: {track_name}. Must be one of {self.TRACK_NAMES}"
            )

        effective_limit = limit if limit is not None else config.limit
        effective_trials = trials if trials is not None else config.trials

        results: list[EvalResult] = []

        if track_name == "prepared_diagnostics":
            result = self._run_prepared_diagnostics(config, effective_limit, endpoint_map)
            results.append(result)

        elif track_name == "tau_episodes":
            results = self._run_tau_episodes(config, effective_limit, effective_trials, endpoint_map)

        elif track_name == "retention":
            result = self._run_retention(config, endpoint_map)
            results.append(result)

        for result in results:
            self._save_result(result)

        return results

    def run_all_tracks(
        self,
        config: EvalRunConfig,
        *,
        limit: int | None = None,
        trials: int | None = None,
        endpoint_map: dict[str, str] | None = None,
    ) -> dict[str, list[EvalResult]]:
        """Run all enabled evaluation tracks.

        Args:
            config: Evaluation run configuration.
            limit: Override number of tasks.
            trials: Override number of trials.
            endpoint_map: Mapping of variant → endpoint URL.

        Returns:
            Dict mapping track name → list of EvalResult.
        """
        all_results: dict[str, list[EvalResult]] = {}

        for track_name in self.TRACK_NAMES:
            track_cfg = self.eval_config.get("tracks", {}).get(track_name, {})
            if not track_cfg.get("enabled", True):
                logger.info("Skipping disabled track: %s", track_name)
                continue

            logger.info("Running track: %s", track_name)
            try:
                results = self.run_track(
                    track_name,
                    config,
                    limit=limit,
                    trials=trials,
                    endpoint_map=endpoint_map,
                )
                all_results[track_name] = results
            except Exception as exc:
                logger.error("Track %s failed: %s", track_name, exc)
                all_results[track_name] = [
                    EvalResult(
                        run_id=str(uuid.uuid4()),
                        track=track_name,
                        execution_status="failed",
                        protocol_deviations=[f"Track execution error: {exc}"],
                    )
                ]

        return all_results

    # -- track runners (private) --------------------------------------------

    def _run_prepared_diagnostics(
        self,
        config: EvalRunConfig,
        limit: int | None,
        endpoint_map: dict[str, str] | None,
    ) -> EvalResult:
        """Run prepared diagnostics for configured variants."""
        ep_map = endpoint_map or {}
        variant = config.variant or "base"
        endpoint_url = ep_map.get(variant, config.model_endpoint)
        if not endpoint_url:
            endpoint_url = _resolve_endpoint_url(variant, self.eval_config)

        return self.diagnostics_evaluator.run(
            endpoint_url,
            variant=variant,
            limit=limit,
            config=config,
        )

    def _run_tau_episodes(
        self,
        config: EvalRunConfig,
        limit: int | None,
        trials: int,
        endpoint_map: dict[str, str] | None,
    ) -> list[EvalResult]:
        """Run τ episodes for all configured variants."""
        tau_config = self.eval_config.get("tracks", {}).get("tau_episodes", {})
        variants = tau_config.get("variants", [])

        if not variants:
            variants = [{"name": config.variant or "base_no_knowledge"}]

        results: list[EvalResult] = []
        for variant_cfg in variants:
            variant_name = variant_cfg.get("name", "")
            endpoint_name = variant_cfg.get("model_endpoint", "base")

            variant_run_config = config.model_copy(
                update={
                    "variant": variant_name,
                    "model_endpoint": (
                        (endpoint_map or {}).get(endpoint_name, "")
                        or _resolve_endpoint_url(endpoint_name, self.eval_config)
                    ),
                    "knowledge_access": variant_cfg.get("knowledge_access", "no_knowledge"),
                    "mode": variant_cfg.get("mode", config.mode),
                    "use_examples": variant_cfg.get("use_examples", config.use_examples),
                }
            )

            logger.info("Running τ episodes for variant: %s", variant_name)
            result = self.episode_runner.run_variant(
                variant_cfg,
                variant_run_config,
                limit=limit,
                trials=trials,
            )

            if result.task_results:
                pe, ci_lo, ci_hi = self.confidence_calculator.compute_ci(result.task_results)
                if result.metrics:
                    result.metrics.ci_lower = ci_lo
                    result.metrics.ci_upper = ci_hi

            results.append(result)

        return results

    def _run_retention(
        self,
        config: EvalRunConfig,
        endpoint_map: dict[str, str] | None,
    ) -> EvalResult:
        """Run retention evaluation."""
        ep_map = endpoint_map or {}
        if not ep_map:
            for name in ("base", "lora", "osft"):
                url = _resolve_endpoint_url(name, self.eval_config)
                if url:
                    ep_map[name] = url

        return self.retention_evaluator.run(ep_map)

    # -- persistence --------------------------------------------------------

    def _save_result(self, result: EvalResult) -> Path:
        """Save an EvalResult to the output directory as JSONL.

        Always saves locally first (before any MLflow logging) so that
        results are never lost.
        """
        track_dir = self._output_dir / result.track
        track_dir.mkdir(parents=True, exist_ok=True)

        summary_path = track_dir / f"{result.variant}_{result.run_id[:8]}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        if result.task_results:
            tasks_path = track_dir / f"{result.variant}_{result.run_id[:8]}_tasks.jsonl"
            _save_jsonl(
                [tr.model_dump(mode="json") for tr in result.task_results],
                tasks_path,
            )

        if result.retention_results:
            retention_path = track_dir / f"{result.variant}_{result.run_id[:8]}_retention.jsonl"
            _save_jsonl(
                [rr.model_dump(mode="json") for rr in result.retention_results],
                retention_path,
            )

        logger.info("Saved evaluation result to %s", summary_path)
        return summary_path

    # -- comparison table ---------------------------------------------------

    def build_comparison_table(
        self,
        all_results: dict[str, list[EvalResult]],
    ) -> list[ComparisonRow]:
        """Build a comparison table across all evaluated variants.

        Args:
            all_results: Output from ``run_all_tracks``.

        Returns:
            List of ComparisonRow for display or MLflow logging.
        """
        rows: list[ComparisonRow] = []

        retention_map: dict[str, RetentionResult] = {}
        for result in all_results.get("retention", []):
            for rr in result.retention_results:
                retention_map[rr.variant] = rr

        for result in all_results.get("tau_episodes", []):
            variant = result.variant
            model_name = variant.split("_")[0] if variant else ""
            ka = result.config.knowledge_access if result.config else ""
            mode = result.config.mode if result.config else ""

            retention_delta = retention_map.get(model_name, RetentionResult(
                benchmark="", variant="",
            )).retention_delta_pp

            m = result.metrics
            rows.append(ComparisonRow(
                variant=variant,
                model=model_name,
                knowledge_access=ka,
                mode=mode,
                task_success_rate=m.task_success_rate if m else None,
                pass_k=m.pass_k if m else None,
                retention_delta_pp=retention_delta or None,
                mean_turns=m.mean_turns if m else None,
                mean_tokens=m.mean_tokens if m else None,
                mean_wall_time=m.mean_wall_time_seconds if m else None,
                failures=m.failed if m else 0,
            ))

        for result in all_results.get("prepared_diagnostics", []):
            m = result.metrics
            if m:
                rows.append(ComparisonRow(
                    variant=result.variant,
                    model=result.variant,
                    diagnostic_accuracy=m.answer_accuracy,
                    mean_turns=m.mean_turns,
                    mean_tokens=m.mean_tokens,
                    mean_wall_time=m.mean_wall_time_seconds,
                    failures=m.failed,
                    notes="τ-Knowledge-derived lab diagnostics",
                ))

        return rows
