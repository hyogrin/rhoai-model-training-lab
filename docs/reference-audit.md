# Reference Audit

This document tracks all external references used by the RHOAI Model Training
Lab, their versions, reused patterns, and verification status.

## References

### 1. τ-bench (τ2-bench)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/sierra-research/tau-bench |
| **Package** | `tau2-bench` (PyPI) |
| **Tested Versions** | `1.0.0`, `1.0.1` |
| **Min Python** | 3.11 |
| **Domain** | `banking_knowledge` |

**Reused Patterns:**
- Tool-discovery interface: `env.get_tools()` → list of tool schemas
- Episode lifecycle: `env.reset()` → initial state, `env.step(tool, args)` → result
- Grading: official scorer/grader interface for reward computation
- User simulator: generates user responses during episodes

**Changes from Original:**
- `TauSimulationAdapter` wraps the τ environment with graceful stub mode
- `TauAgentBridge` translates τ `Agent`/`LLM` interface to `/v1/agent/step` HTTP calls
- `filter_private_fields()` strips evaluator-only data before model exposure
- Version compatibility checks in `check_tau_compatibility()`

**Verification Status:** ✅ Version pinning validated; stub mode tested offline

---

### 2. sdg_hub

| Field | Value |
|-------|-------|
| **URL** | Internal package |
| **Package** | `sdg_hub` (PyPI) |
| **Version** | `>=0.1` |
| **Import** | `import sdg_hub` |

**Reused Patterns:**
- `sdg_hub.create_pipeline(endpoint, api_key, model, ...)` → generator callable
- Generator callable: `fn(sample_type, flow, max_tokens, temperature, seed)` → raw output dict
- Pipeline returns `{"messages": [...], "tools": [...], "source_doc_ids": [...], "scenario_family": "...", "cost_usd": float}`

**Changes from Original:**
- Wrapped in `SDGPipeline` with budget tracking, checkpointing, and validation
- `PolicyExtractor` pre-processes KB documents into policy clauses
- `SyntheticValidator` performs multi-dimensional validation (schema, grounding, dedup, contamination)
- Cost accumulation with budget limit enforcement

**Verification Status:** ⚠️ Requires `sdg_hub` installation for live testing; pipeline logic tested with mocks

---

### 3. training_hub

| Field | Value |
|-------|-------|
| **URL** | Internal package |
| **Package** | `training_hub` (PyPI) |
| **Version** | `>=0.1` |
| **Import** | `import training_hub` |

**Reused Patterns:**
- `training_hub.lora_sft.train(config)` → result dict with training metrics
- `training_hub.osft.train(config)` → result dict with training metrics
- Config format: `{"model": {...}, "lora"/"osft": {...}, "data": {...}, "training_args": {...}}`

**Changes from Original:**
- `LoRATrainer` and `OSFTTrainer` wrap hub calls with lifecycle management
- Native fallback via `transformers.Trainer` + `peft` when `training_hub` is unavailable
- `DataPreprocessor` handles chat-template application and loss masking verification
- `validate_bundle_for_training()` performs pre-training readiness checks

**Verification Status:** ⚠️ Requires `training_hub` installation; native fallback path tested

---

### 4. micro-financial-loan (Reference Architecture)

| Field | Value |
|-------|-------|
| **URL** | Internal reference architecture |
| **Reference** | RHOAI 3.5 deployment patterns |

**Reused Patterns:**
- KServe `InferenceService` manifest structure for vLLM
- `RawDeployment` mode with GPU resource requests
- vLLM CLI arguments: `--model`, `--served-model-name`, `--dtype`, `--tool-parser-plugin`
- Hermes tool-parser configuration for structured tool calls
- OpenShift Route with TLS edge termination

**Changes from Original:**
- Adapted for `Qwen3-4B-Instruct-2507` (different model size / architecture)
- Added `--enable-auto-tool-choice` for automatic tool selection
- `--gpu-memory-utilization` configurable per deployment
- `ServingManifestRenderer` generates manifests from Jinja2 templates

**Verification Status:** ✅ Manifest structure validated against KServe v1beta1 API schema

---

### 5. rhoai-custom-research-lab (Reference Lab)

| Field | Value |
|-------|-------|
| **URL** | Internal lab reference |
| **Reference** | Lab notebook and infrastructure patterns |

**Reused Patterns:**
- Notebook structure: setup → data loading → training → evaluation → results
- Environment profile system (preparation, lora, osft, backend, evaluation)
- Preflight checks CLI command
- MLflow experiment naming convention
- Bundle-based data distribution (local → S3 → PVC fallback chain)

**Changes from Original:**
- τ-Knowledge domain instead of generic research tasks
- Three-track evaluation (diagnostics, τ episodes, retention) vs single-track
- pass^k semantics (consistent success) instead of pass@k (best-of-k)
- Paired bootstrap CIs for variant comparison
- `BundleBuilder` with policy-aware splitting and validation

**Verification Status:** ✅ Patterns verified against reference lab documentation

---

### 6. run_eval.py (τ Evaluation Runner)

| Field | Value |
|-------|-------|
| **URL** | `tau-bench/run_eval.py` in τ-bench repository |
| **SHA** | Pinned to τ2-bench release version |

**Reused Patterns:**
- Episode execution loop: reset → inference → tool execution → grading
- Budget enforcement: max_turns, max_tool_calls, max_wall_time
- Task-level success aggregation with trial repetitions
- Grader interface: `grade(task_id, session_id, messages, actions)` → `{success, reward}`

**Changes from Original:**
- `TauEpisodeRunner` class wraps the loop with resume support and fingerprint deduplication
- `EvalOrchestrator` manages all three tracks and comparison table generation
- Confidence intervals computed via `ConfidenceCalculator` (paired bootstrap)
- Results saved via `LocalResultStore` before MLflow logging
- Adapter hooks (`_reset_environment`, `_execute_tool`, `_grade_episode`) for testability

**Verification Status:** ⚠️ Offline evaluation logic tested; live τ-bench integration requires environment setup

---

## Verification Summary

| Reference | Status | Notes |
|-----------|--------|-------|
| τ-bench | ✅ Verified | Version pinning + stub mode |
| sdg_hub | ⚠️ Partial | Mock-tested; needs live endpoint |
| training_hub | ⚠️ Partial | Native fallback tested |
| micro-financial-loan | ✅ Verified | Manifest schema validated |
| rhoai-custom-research-lab | ✅ Verified | Pattern alignment confirmed |
| run_eval.py | ⚠️ Partial | Offline logic verified |
