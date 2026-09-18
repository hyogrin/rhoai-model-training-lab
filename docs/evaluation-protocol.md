# Evaluation Protocol

## Overview

The RHOAI Model Training Lab evaluates model variants across **three tracks**,
each targeting a different aspect of model quality. All results are logged to
MLflow with full lineage and reproducibility metadata.

## Three Evaluation Tracks

### Track A: Prepared Diagnostics (`prepared_diagnostics`)

**Purpose:** Offline evaluation on holdout samples derived from τ-Knowledge
banking data. Results are labelled as *τ-Knowledge-derived lab diagnostics*,
not official τ pass rates.

**Method:**
1. Load holdout validation samples from the prepared bundle
2. For each sample, send the question (with matched context) to the model endpoint
3. Score on five dimensions:
   - **Answer accuracy** — exact/normalised substring match
   - **Condition accuracy** — Jaccard overlap of identified conditions
   - **Tool name accuracy** — binary match of predicted vs expected tool
   - **Tool argument accuracy** — key-level argument match
   - **Grounding score** — recall of expected document references

**Matched Context Mode:** All model variants (base, LoRA, OSFT) receive
identical KB excerpts, ensuring fair comparison independent of retrieval quality.

---

### Track B: τ Episodes (`tau_episodes`)

**Purpose:** End-to-end evaluation using official τ-Knowledge episode execution.
The model interacts with a simulated banking customer via tool calls.

**Method:**
1. For each task × trial combination:
   - Reset the τ simulation environment
   - Run an agent loop: model inference → tool execution → observation
   - Grade the episode using the official τ grader
2. Aggregate results using **pass^k** (see below)
3. Compute paired bootstrap confidence intervals

**Episode Budget Enforcement:**
- `max_turns`: 20 (default)
- `max_tool_calls`: 15
- `max_tokens`: 8192
- `max_wall_time_seconds`: 300

**Timeout/failure handling:** Episodes that time out or exceed budget are
recorded as failures. They are **always kept in the denominator** — never
excluded from success rate calculations.

---

### Track C: Retention (`retention`)

**Purpose:** Verify that fine-tuned models preserve general capability
on a non-banking benchmark (ARC-Challenge).

**Method:**
1. Load ARC-Challenge test samples (100 by default, reproducibly sampled)
2. Send each as a zero-shot generated-answer prompt (no banking system prompt)
3. Compare predicted answer letter to the correct answer
4. Compute `retention_delta_pp = 100 × (adapted_accuracy - base_accuracy)`

**Acceptance criteria:** `retention_delta_pp ≥ -2.0` (no more than 2 pp
regression from the base model).

---

## Variant Matrix

Each track evaluates up to 6 variants, crossing 3 models with 2 knowledge modes:

| Variant Name | Model | Knowledge Access |
|-------------|-------|-----------------|
| `base_no_knowledge` | Base (Qwen3-4B) | None |
| `base_rag` | Base (Qwen3-4B) | RAG |
| `lora_no_knowledge` | LoRA-merged | None |
| `lora_rag` | LoRA-merged | RAG |
| `osft_no_knowledge` | OSFT-exported | None |
| `osft_rag` | OSFT-exported | RAG |

For retention (Track C), only the no-knowledge variants are evaluated
(base, lora, osft) since the benchmark is domain-independent.

---

## pass^k Semantics

This lab uses **pass^k**, not pass@k.

### pass^k (consistent success)

```
pass^k = Σ_task [ ∏_{i=1}^{k} success_i ] / |tasks|
```

A task is counted as a pass^k success **only if the model succeeds on
ALL k trials**. A single failure on any trial makes that task score 0.

### Why not pass@k?

pass@k (from code generation) measures best-of-k: "did the model get it right
at least once in k tries?" This is useful for sampling-based code generation
but inappropriate for banking agents, where **consistent, reliable** behavior
matters. A banking system that occasionally gives wrong answers is unsafe,
even if it sometimes gives correct ones.

### Example

| Task | Trial 1 | Trial 2 | Trial 3 | pass^3 | pass@3 |
|------|---------|---------|---------|--------|--------|
| A | ✓ | ✓ | ✓ | 1 | 1 |
| B | ✓ | ✗ | ✓ | **0** | 1 |
| C | ✗ | ✗ | ✗ | 0 | 0 |

- pass^3 = 1/3 = 0.333
- pass@3 = 2/3 = 0.667

---

## Confidence Intervals

Confidence intervals are computed using **paired bootstrap** resampling
at the task level:

1. Group results by task ID
2. For each bootstrap iteration (default: 10,000):
   - Resample tasks (with replacement)
   - Compute the metric (success rate or pass^k)
3. Take the α/2 and (1 - α/2) percentiles of the bootstrap distribution
4. Default α = 0.05 (95% CI)

The paired bootstrap preserves the **dependence structure** between repeated
trials of the same task and between variants evaluated on the same task set.

For variant comparison, paired CIs on the difference (rate_A - rate_B) are
computed by resampling common tasks and computing per-task differences.

---

## Protocol Deviation Reporting

Any deviation from the standard evaluation protocol is recorded in the
`protocol_deviations` field of the `EvalResult`:

- Missing task IDs (tasks not attempted)
- Budget limit overrides
- Non-standard grader or scorer versions
- Interrupted evaluation runs (partial results)
- Modified simulation parameters

Deviations are surfaced in MLflow tags and the comparison table `notes` column.

---

## Verification Levels

| Level | Meaning |
|-------|---------|
| `unverified` | Default — results not yet confirmed |
| `fixture_verified` | Validated against known success/failure trajectories |
| `live_verified` | Validated against live τ-bench simulation |

The `verification_status` field in `EvalResult` tracks the current level.
Fixture trajectories (in `tests/fixtures/`) provide deterministic baseline
validation.
