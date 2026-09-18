# rhoai-model-training-lab — τ-Knowledge Implementation Specification and Opus Prompts

Revision date: 2026-09-17. Revision 3: English implementation specification.
Target platform: Red Hat OpenShift AI Self-Managed 3.5.x.

This document specifies a repository to be implemented. Updating this specification does not mean that synthetic data generation, GPU training, or cluster deployment has already been performed.

## 0. Approved Decisions and How to Use This Document

The selected domain is **τ-Knowledge `banking_knowledge`**. The system must handle banking customer requests by consulting knowledge documents and current account observations, then answering or performing policy-compliant operations through simulation tools.

**Complete data preparation first, then keep training notebooks simple.** An operator or instructor runs synthetic data generation, validation, splitting, and backend-specific conversion once, and distributes a versioned **prepared dataset bundle**. Learners must be able to train LoRA and OSFT models using this bundle without access to a teacher API and without rerunning SDG. Source collection and SDG notebooks remain available as a separate authoring and advanced-learning path.

| Stage | Workflow | Deliverables |
|---|---|---|
| 1A. Prepare data in advance | Pin sources → generate synthetic data → independently validate → split → export → release bundle | Training-ready JSONL, manifests, quality reports, configurations |
| 1B. Learner training | Download and validate bundle → train LoRA and OSFT independently → export and deploy | Model checkpoints and serving endpoints |
| 2. RAG and harness | Index the shared KB and training examples → implement planning and tool execution | Backend, startup scripts, official simulation adapter |
| 3. Evaluation | Prepared offline diagnostics + official τ-Knowledge episodes + capability retention | MLflow experiments and comparison notebooks |

Give Opus this entire document and the master prompt in Section 10. Use Sections 11–14 for staged implementation and Section 15 for final review. If a repository already exists, preserve user changes and resume from `docs/implementation-status.md`.

## 1. Scope and Baseline Architecture

- Start from an existing RHOAI Workbench. Cluster and Workbench installation are prerequisites, not implementation tasks.
- The initial model candidate is `Qwen/Qwen3-4B-Instruct-2507`. Verify common compatibility across LoRA, OSFT, and tool-capable serving before pinning its revision. A smaller Qwen may be used for smoke tests; do not promise that it will perform well on full agent tasks.
- Use `sdg_hub` in the actual synthetic generation path and `training_hub` in the actual LoRA and OSFT training paths. Importing these packages while bypassing them is not acceptable.
- Start both training branches independently from the same base checkpoint and the same logical training samples. Do not apply OSFT to the LoRA-trained checkpoint.
- Use text interfaces. Voice, a frontend, Dify, mandatory OpenShell or MCP services, and a separate production vector database are out of scope.
- Use FastAPI, a small LangGraph harness, a local persistent vector index, and an adapter to the official τ simulation. Connect business tools to their official simulation implementations.
- Use separately configured endpoints for the SDG teacher and evaluation user simulator. Neither endpoint may be required for learner training.
- This implementation specification and its coding prompts are written in English. Preserve the existing learner-facing language requirement: notebook explanations and learner documentation are in Korean; code identifiers, configuration keys, and log keys are in English. Keep benchmark and primary training content in its original English. Translated datasets constitute separate experiments.

## 2. References and Implementation Audit

### 2.1 τ-Knowledge and τ-bench

The [τ-Knowledge paper](https://arxiv.org/abs/2603.04370), released on March 4, 2026, evaluates banking assistance that combines unstructured knowledge with tool-mediated operations. The [official repository](https://github.com/sierra-research/tau2-bench) retains the path `tau2-bench`, while its reviewed README describes τ³-bench and `banking_knowledge`. Pin a release and commit SHA; do not infer the version from the repository name.

At review time, the README documented a July 2026 v1.0.1 banking grading correction and warned that affected scores before and after the correction are not comparable. Prefer a version containing that correction after checking compatibility, and record both task and scorer revisions. Verify Python requirements. The τ evaluator environment may be separate from the RHOAI training environments.

The [Knowledge Retrieval documentation](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/knowledge/README.md) describes `no_knowledge`, `full_kb`, `golden_retrieval`, BM25, and dense retrieval configurations. Reserve `golden_retrieval` for explicitly labeled oracle diagnostics. Inspect actual task schemas, reward computation, tool discovery, simulator access boundaries, and task splits before implementation. Validate configuration names against the pinned version.

### 2.2 Roles of the Existing References

| Reference | Reuse | Do not copy or assume |
|---|---|---|
| [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) | YAML flows, structured generation, filtering, checkpointing | Unverified built-in flows or API signatures |
| [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) | Actual `lora_sft`, `osft`, and export examples | Equality between upstream and product-packaged versions |
| [micro-financial-loan](https://github.com/cbtham/micro-financial-loan/tree/main) | LoRA merge, Hugging Face artifact export, S3 upload, vLLM deployment patterns | Hardcoded secrets or cluster addresses, disabled TLS verification, assumed training times |
| [rhoai-custom-research-lab](https://github.com/hyogrin/rhoai-custom-research-lab) | Graph/state, context, tools, observability layers | Mandatory adoption of its entire UI, database, and MCP stack |
| [run_eval.py](https://github.com/hyogrin/rhoai-inference-design-planner/blob/main/scripts/run_eval.py) | CLI, result files, MLflow parameters, metrics, artifacts | Its 100-point golden-text similarity rubric |
| [RHOAI 3.5 documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/release_notes/index) | Actual Workbench, serving, and MLflow configuration | Universal package availability or automatic tracking of all custom code |

The reference `run_eval.py` scores previously saved responses. This repository must additionally call live endpoints, execute τ episodes, and collect official rewards. Text similarity is not business-task success.

Record actual source paths, URLs, SHAs, signatures, reused patterns, changes, and verification status in `docs/reference-audit.md`. For the harness reference, inspect `agents/orchestrator/graph.py`, `state.py`, `layers/`, `backend/api.py`, and `backend/observability.py`. Distinguish product support, upstream functionality, and observed success in the current environment.

## 3. First Deliverable: A Prepared Dataset Bundle

### 3.1 Learner and Authoring Paths

The learner quickstart must require only the following target commands before training. These are interfaces to implement in this lab, not claims about existing upstream APIs.

```bash
bash scripts/setup_env.sh --profile lora
python scripts/fetch_prepared_dataset.py --release configs/data-release.yaml
python scripts/validate_prepared_dataset.py --bundle data/prepared/tau-knowledge-v1
# Then run notebook 03 or 04 in its configured environment.
```

Provide a separate operator path:

```bash
python scripts/prepare_tau_sources.py --config configs/data-preparation.yaml
python scripts/generate_synthetic.py --config configs/sdg.yaml
python scripts/validate_synthetic.py --config configs/data-preparation.yaml
python scripts/build_prepared_bundle.py --config configs/data-release.yaml
```

A failed download must not silently trigger SDG. Report missing files, checksum mismatches, licensing conditions, and version incompatibilities explicitly. A disconnected Workbench must be able to load the same bundle from a PVC or accessible S3-compatible storage.

### 3.2 Bundle Contents

| Path | Contents |
|---|---|
| `manifest.json` | Bundle version, actual τ SHA and task revision, source hashes, model/tokenizer profile, counts, split policy |
| `checksums.sha256` | Integrity checksums for bundle payload files |
| `canonical/train.jsonl`, `validation.jsonl` | Validated common chat/tool examples using a model-independent schema |
| `training/lora/train.jsonl`, `validation.jsonl` | Files loadable by the pinned LoRA backend |
| `training/osft/train.jsonl`, `validation.jsonl` | Files loadable by the pinned OSFT backend |
| `configs/lora.yaml`, `osft.yaml` | Data paths and training profiles, with documented hardware overrides |
| `kb/documents.jsonl` | Redistributable KB snapshot, document IDs, text, links, provenance |
| `examples/train_examples.jsonl` | Training-only retrieval examples with provenance |
| `metadata/provenance.jsonl` | Canonical IDs, source documents, scenario families, generator and validator identities, outcomes |
| `metadata/splits.json` | Task/scenario-family partitions and official holdout IDs or their hashes |
| `reports/quality.json`, `quality.md` | Acceptance rates by type, rejection reasons, token statistics, coverage, review results |
| `reports/backend-validation.json` | Loader/tokenizer checks for both backends and the actual verification scope |
| `DATASET_CARD.md`, `LICENSES/` | Generation, validation, split methodology, limitations, redistribution conditions |

Do not include evaluation-private instructions, expected actions, or reward criteria in the training/RAG bundle. Keep them in a separate evaluation bundle or official installation path accessible only to the evaluator. Distinguish user-visible messages from evaluator-private user scenarios.

A prepared bundle must contain actual validated data bytes and checksums, not just scripts capable of producing them. Include a small smoke bundle in the repository if redistribution is permitted. Store larger bundles at versioned release-artifact, S3, or OCI locations. If remote upload access is unavailable, finish the local bundle and provide exact publishing instructions. Never invent download URLs or checksums.

Initial size targets: 64 accepted samples for smoke testing and 2,000–5,000 accepted samples for the lab profile. These are targets, not guaranteed yields. Finalize profiles using actual accepted counts and coverage. Do not inflate the dataset with low-quality samples to meet a count.

## 4. SDG Design: Validation and Training Usability First

### 4.1 Sources and Fairness

- Use the KB snapshot, public agent policies, tool schemas, and independently created training simulation states.
- Official training tasks may be used as seeds only if an appropriate training split exists and its terms permit that use. Do not assume the existence of train/dev/test splits before inspecting the pinned release.
- If there is no suitable official training split, reserve all official tasks for evaluation and generate synthetic train/dev data from the KB and independent scenario generators.
- Label experiments that train on the public KB as `kb_adaptation`. They measure learning a KB and applying it to new workflow situations, not generalization to unseen documents. Do not claim equivalence to a training-free leaderboard protocol.
- Do not use evaluation task wording, hidden scenarios, expected action sequences, golden document lists, or task-specific evaluator states to generate training scenarios.
- Sharing the KB between training and retrieval is intentional in `kb_adaptation`; leaking tasks, scenarios, or answers is not. Do not introduce an unconditional document-disjoint requirement that contradicts this experiment.
- In an optional `unseen_policy` experiment, partition policy/document families before SDG and provide equal evidence access to all evaluated models.

### 4.2 Training Example Types

| Type | Input → Target | Validation |
|---|---|---|
| Policy knowledge QA | Question → accurate conditions and exceptions | Source, conditions, limits, and units |
| Context-grounded policy application | Documents + customer situation → allowed, disallowed, or clarification needed | Condition truth tables or reviewed structured rules |
| Clarification | Incomplete situation → necessary follow-up question | Correspondence to genuinely missing fields |
| Tool selection and arguments | Current observations + available tools → next tool call | Schema, entities, and state consistency |
| Multi-step trajectories | Conversation and tool observations → assistant actions and responses | Replay in the official simulation; policy and final-state checks |
| Exceptions, conflicts, insufficient evidence | Limited evidence → clarification, further retrieval, or appropriate termination | No unsupported success claims or invented conditions |

Do not train only on simple QA and assume that official agent-task performance will improve. The lab bundle must include validated tool-use examples and short trajectories. Apply separate quality gates to the more difficult multi-step examples and report actual accepted counts.

Suggested initial distribution: policy QA 30%, policy application 25%, tool selection 20%, trajectories 15%, clarification/exceptions 10%. Finalize this distribution within the generator and validator's verified capabilities and record it in the manifest. It is not the official benchmark distribution.

### 4.3 SDG Hub Pipeline

1. Ingest the KB while preserving document boundaries, sections, links, policy versions, conditions, and exceptions.
2. Extract structured policy facts and conditions. Retain source offsets and document IDs; review a sample of extraction results.
3. Assign scenario templates/families to train or validation before generation. Paraphrases, name/number substitutions, and sibling examples of the same workflow must remain in one split.
4. Have the teacher generate scenarios, responses, and required tool calls against an explicit JSON schema. Use independent simulation fixtures rather than real customer/account IDs.
5. Validate schemas, grounding, conditions, units, tool names and arguments, conversation order, and citations.
6. Replay tool examples through the official simulation adapter to verify preconditions and resulting states. A generator's success assertion is not sufficient.
7. Run an independent validator and sampled human review. LLM judging alone does not constitute complete verification. Record whether generator and validator share a model.
8. Check exact/near duplicates and scenario-family leakage. Run evaluation-contamination checks in an isolated evaluator process and return only minimal pass/fail and hash-based findings.
9. Normalize accepted data into the canonical schema and pre-export backend-specific training files.
10. Validate every accepted sample using both backend loaders and the pinned tokenizer before releasing the bundle.

Support budgets, bounded concurrency, retry/backoff, checkpoints, resume, rejection logs, and teacher usage accounting. Quarantine rejected data. Do not request or store lengthy private reasoning traces; use concise plans, evidence, and observable actions as training targets.

### 4.4 Backend-Ready Format and Loss Masking

Canonical records contain `sample_id`, `messages`, optional `tools`, `source_doc_ids`, `scenario_family`, and `validation_status`. Remove provenance fields from model inputs and retain them in sidecar metadata.

```json
{
  "sample_id": "synthetic-policy-0001",
  "messages": [
    {"role": "system", "content": "You are a banking support assistant."},
    {"role": "user", "content": "<synthetic request and permitted observations>"},
    {"role": "assistant", "content": "<validated response>"}
  ]
}
```

This is a schema illustration, not training data. Convert tool-call examples only after verifying the selected model's chat template and each backend's accepted representation. Do not assume that every backend directly accepts OpenAI-style tool messages.

- Maintain a one-to-one mapping of canonical IDs to LoRA and OSFT exports. Check omissions and transformations and save the mapping and hashes.
- Apply loss to assistant text and assistant tool-call targets only. Mask system messages, user messages, tool observations, and padding. If a backend does not support this, implement a verified preprocessing path or mark the profile blocked.
- Render tool schemas and serialize assistant calls consistently with inference. Do not omit tool schemas required to interpret a training target.
- Apply the chat template exactly once. Check BOS/EOS/padding, assistant termination, tool-call ID references, and ordering.
- Use role-message JSONL by default. Token caches are optional and must be rebuilt when tokenizer revision or template hash changes.
- Validate maximum lengths and token distributions per model/tokenizer profile. Shorten long tool observations without losing necessary evidence/state, or exclude the example during preparation. Do not silently truncate them in a training notebook.
- Backend storage formats may differ, but logical samples and loss-target semantics must match. Report a comparison limitation if equivalence cannot be maintained.

## 5. Simple Training Notebooks and Deployment

### 5.1 Notebook Paths

| Audience | Notebook | Purpose |
|---|---|---|
| Learner | `00_preflight.ipynb` | GPU, environment, PVC, MLflow, model profile |
| Learner | `01_load_prepared_dataset.ipynb` | Fetch bundle, validate checksums and compatibility, inspect examples/statistics |
| Learner | `03_lora_finetuning.ipynb` | Train and save using prepared files |
| Learner | `04_osft_finetuning.ipynb` | Independently train and save using the same base/data |
| Learner | `05_export_and_deploy.ipynb` | LoRA merge/OSFT export, S3 transfer, serving smoke tests |
| Learner | `06_rag_harness.ipynb` | Indexing, planning, tool adapter, API walkthrough |
| Learner | `07_evaluate.ipynb` | Offline diagnostics, episodes, and MLflow |
| Learner | `08_compare_results.ipynb` | Retrieve and compare experiment results |
| Author/advanced | `data_preparation/01_prepare_tau_sources.ipynb` | Pin sources and explain KB/task boundaries |
| Author/advanced | `data_preparation/02_generate_synthetic.ipynb` | Explain and execute SDG Hub flows |
| Author/advanced | `data_preparation/03_validate_and_release.ipynb` | Quality validation, backend export, bundle release |

Target approximately **5–7 executable cells** in each of notebooks 03 and 04: configure → validate bundle → preview examples/loss targets → train → inspect results/MLflow → reload checkpoint or proceed to export. Include enough explanatory Markdown for the learner.

Training notebooks must not download sources, call a teacher, remove duplicates, create splits, perform complex tool-format conversion, or require manual data repair. Fail with a bundle-validation error when data is invalid; link to the authoring path rather than silently regenerating it.

For educational transparency, show the actual `training_hub.lora_sft` / `osft` invocation and important parameters in a short cell or linked source. Isolate compatibility details in a thin adapter. Do not build a large custom framework that obscures training.

### 5.2 Training Conditions

- Pin identical base revision, tokenizer profile, bundle ID, canonical sample IDs, and seed across training branches.
- LoRA rank 16/alpha 32 and OSFT unfreeze rank ratio 0.25 are initial candidates, to be verified and selected using development data. Do not force identical learning rates across different algorithms.
- Log dataset size, tuning budget, actual tokens/steps, wall time, and peak VRAM. Equal epoch counts do not imply equal compute.
- Account for OSFT decomposition, activations, and optimizer memory. Do not estimate training resources from weight size alone.
- Separate SDG, LoRA, OSFT, and τ evaluation environments/kernels when needed. Avoid repeatedly overwriting torch/CUDA dependencies. Provide reproducible locks or constraints.
- Run training in an isolated process where appropriate and use the GPU sequentially. Support checkpoint resume and failure logs.
- Better data preparation does not eliminate hardware or model-capacity constraints. Do not promise full-benchmark success rates for a small model.
- Full SFT is an optional control. Do not make a causal claim that OSFT reduces forgetting relative to ordinary full SFT without that control.

### 5.3 Deployment on RHOAI 3.5

Inspect installed CRDs, runtime templates, and the official 3.5 documentation before implementing single-model vLLM deployment. MaaS and llm-d are optional. Record runtime image digest, model revision, chat template, and tool-parser settings.

Preserve the original LoRA adapter and separately export the merged Hugging Face model, tokenizer, and configuration. Export OSFT checkpoints to a serving-compatible Hugging Face artifact and validate reloading. Check shards, index files, hashes, and tool-call responses. Use S3-compatible storage as the default transfer path; ModelCar is optional. Distinguish model weights from runtime images.

Do not copy hardcoded credentials or disabled TLS verification. Use endpoint-specific secrets, CA configuration, and settings. Without deployment access, complete manifest rendering and exact commands and report the unexecuted apply step. Provide namespace-scoped cleanup, readiness checks, and authenticated chat/tool-call smoke tests.

Support sequential serving of Base, LoRA, and OSFT on one GPU. If the same endpoint URL is reused, verify the actual model artifact hash before each evaluation. Check `/v1` normalization, served model name, and tool-call parsing during preflight.

## 6. RAG and Planning Harness

### 6.1 Boundaries Between Knowledge, State, and Evaluation Information

| Data | Purpose | Agent access |
|---|---|---|
| KB snapshot | Policies, products, procedures | Retrieval or context, according to the experiment |
| Synthetic training examples | Examples of reasoning patterns and tool use | Optional training-only example index |
| Current simulator database | Account state for the current episode | Only permitted observations through official tools |
| Private task criteria | Success conditions, expected actions, hidden user goals | Evaluator only |

Do not inject the entire business database into prompts or a vector index. Do not forward user-simulator private instructions as assistant input. Preserve official tool-discovery behavior. If a convenience mode exposes all tool schemas upfront, label it as a modified benchmark condition.

### 6.2 Runtime Structure

Use FastAPI, a small LangGraph state graph, and a local persistent vector store. Retain BM25 as a diagnostic baseline and build dense indexes using a pinned embedding model. Support a local CPU embedding model or a configured endpoint. Implement an adapter rather than assuming an upstream Qwen embedding configuration automatically uses RHOAI.

Separate KB and example collections. Examples demonstrate methods; they are not facts about the current account. Include corpus, embedding revision/dimension, and chunking version in the index fingerprint. Preserve policy exceptions and document links when chunking.

Graph: receive observations → create a concise structured plan → retrieve knowledge/discover tools → apply policy → issue an authorized business tool call → inspect and verify results → ask a necessary question or continue → finish or fail.

- Planning must affect queries, tool selection, or execution branches.
- Enforce limits on turns, tool calls, repairs, tokens, and wall time. Wait for a real user or official user simulator to answer a clarification. Do not fabricate the answer.
- The verifier may use only current permitted observations and public policies, never hidden expected actions or states.
- Do not blindly retry state-changing tool calls. Use official semantics and observed results to prevent duplicate operations.
- Reset the environment for every task/trial and isolate session state. Parallel evaluations must not share mutable databases or simulator state.
- `simple_rag` performs fixed retrieval per dialogue turn followed by a response or next action, without additional planning/repair. `agent_rag` uses the same model, retriever, and business tools while allowing explicit planning and additional retrieval.
- The default implementation is simulation-only. Do not connect it to real banking accounts or external business systems.

### 6.3 API and Startup Scripts

Provide `GET /healthz`, `GET /readyz`, `POST /v1/agent/step`, and `POST /v1/qa`.

`/v1/agent/step` accepts session/request IDs, permitted message history, tool observations, currently exposed schemas, and a mode. It returns the next assistant message or tool calls. A clearly designated simulation adapter owns tool execution and state updates. Do not create a path that uses a hidden task ID to look up the expected answer.

`/v1/qa` is for prepared knowledge-QA diagnostics and does not replace official business-task evaluation. Responses include status, content/tool calls, citations where applicable, model/artifact/corpus hashes, usage, and trace IDs. Do not disguise 401/403 errors or timeouts as successful responses.

```bash
bash scripts/setup_env.sh --profile backend
bash scripts/build_index.sh --bundle data/prepared/tau-knowledge-v1
bash scripts/start_backend.sh --host 127.0.0.1 --port 8000
bash scripts/smoke_backend.sh
bash scripts/stop_backend.sh
```

Default to foreground execution. Optional background execution must manage PID/log files, detect port conflicts, and terminate only the process started by the script. Do not require sudo, systemd, or Docker-in-Docker. Provide a Containerfile and Deployment/Service/Route/PVC/Secret examples covering arbitrary UIDs, write permissions, probes, and resources. Shell-search sandboxing is optional.

## 7. Evaluation: Separate Simple Diagnostics From Official Task Success

### 7.1 Three Evaluation Tracks

**A. `prepared_diagnostics` — easy-to-understand notebook evaluation**

Use independently generated and reviewed holdout QA, policy-application, and next-action examples. Separate scenario families from training and restrict labels to the evaluator. Score answers/conditions, tool names and arguments, grounding, and clarification requirements. Use deterministic checks where possible; otherwise use an explicit rubric, pinned judge, and sampled review.

Label these results **τ-Knowledge-derived lab diagnostics**. Synthetic single-turn accuracy is not an official τ pass rate. In the `matched_context` condition, give Base, LoRA, and OSFT identical KB excerpts and observations.

**B. `tau_episodes` — end-to-end evaluation in the official environment**

The official runner, or a semantically equivalent runner adapter, owns the user simulator, tools, environment resets, termination, and grading. Every assistant inference turn must call the actual deployed model endpoint. When using the lab backend, bridge the official agent interface to `/v1/agent/step`. Do not infer task success solely from a final text answer.

Use a model-by-knowledge-access design under the same controller and business-tool environment:

| Variant | Model | Knowledge access |
|---|---|---|
| `base_no_knowledge` | Base | No KB retrieval |
| `lora_no_knowledge` | LoRA | No KB retrieval |
| `osft_no_knowledge` | OSFT | No KB retrieval |
| `base_rag` | Base | Shared RAG configuration |
| `lora_rag` | LoRA | Shared RAG configuration |
| `osft_rag` | OSFT | Shared RAG configuration |

Keep current-account observations, business tools, and simulator conditions identical across groups. `no_knowledge` does not mean removing business tools. Report the KB scope and actual training coverage for `kb_adaptation` experiments.

Evaluate planning separately with `base_simple_rag` versus `base_agent_rag`. Evaluate example retrieval with `use_examples=false/true` under otherwise identical conditions. Do not attribute changes in controller, examples, or tool access to model training.

Pin task IDs, trial counts, simulator model/revision/prompt/seed, budgets, model sampling, and retriever configuration. Do not repeatedly tune on public evaluation tasks; use separate synthetic development data. Changes to official tasks, splits, or policies must be reported as a modified protocol.

τ's `pass^k` measures consistent success over repeated attempts; it is different from code-generation best-of-k `pass@k`. Use the estimator from the pinned official evaluator. Do not calculate a metric when there are too few trials. Action recall is supplementary: it can overlook unnecessary or wrong actions and must not replace task success.

**C. `retention` — preservation of existing capabilities**

Disable KB/RAG and banking system prompts. Evaluate Base, LoRA, and OSFT on identical samples from a separate fixed benchmark. Default to ARC-Challenge generated-answer accuracy, with IFEval as an optional extension after verifying its official implementation. Do not conflate generation-based accuracy with log-likelihood leaderboard scores. Record `retention_delta_pp = 100 × (adapted_accuracy - base_accuracy)`. A small single-benchmark sample cannot establish elimination of forgetting.

### 7.2 Official Evaluation Adapter and Error Handling

- Inspect the installed τ Agent/LLM interfaces, message/tool schemas, reward basis, grader, and actual split names. Test that private evaluator fields never enter model payloads.
- Reuse official evaluation criteria. Validate the adapter with known-success and known-failure trajectory fixtures. Do not substitute golden-text similarity for the grader.
- Implement authentication, CA verification, URL normalization, timeouts, bounded retries for 429/transient 5xx responses, concurrency limits, and request/response storage with secret redaction.
- Fingerprint runs using task ID, trial, model hash, data revision, and configuration hash. Resume an episode only from a valid saved simulation state; otherwise restart it from the initial state. Replaying messages alone is insufficient.
- Keep failures, timeouts, invalid tool calls, and budget exhaustion in the denominator. Report attempted, completed, succeeded, failed, and unsupported counts.
- Separate evaluator-judge and user-simulator costs from agent-model inference costs.
- Report episode wall time, total agent tokens/calls, retrieval/tool latency, and success rates. Do not infer TTFT from non-streaming latency.
- Provide task-level paired bootstrap confidence intervals while preserving dependence between repeated trials. A smoke run of approximately five episodes and one trial validates the path; it is not a benchmark performance conclusion.
- Explicitly report protocol deviations, including workarounds for model tool-use limitations.

### 7.3 MLflow

Use experiments named `rhoai-model-training-lab-data`, `rhoai-model-training-lab-training`, and `rhoai-model-training-lab-evaluation`. Link data-release runs → training runs → evaluation suite/variant runs with IDs and hashes.

| Category | Required records |
|---|---|
| Data | KB/τ revision, generator/validator configuration, acceptance/rejection/coverage, bundle ID/hash, review status |
| Training | Base/tokenizer, canonical/export hashes, seed, loss, actual steps/tokens, duration/VRAM |
| Evaluation tags | Track, protocol, variant, execution mode, verified/not_run/blocked |
| Evaluation configuration | Task/split/revision, grader, user simulator, model artifact, controller, retriever, budgets |
| Metrics | Official task success/pass^k when available, diagnostic accuracy, action recall, errors, latency/tokens/calls, retention |
| Artifacts | Redacted configuration, manifests, per-task/per-trial JSONL, failures, traces, summaries/comparisons, plots, scorer identity |

Do not report successful MLflow persistence when logging fails. Save local results first and record the failure. `--no-mlflow` is an explicit offline mode. Provide fingerprint-based re-upload/deduplication through `scripts/log_eval_results.py`. Do not automatically upload multi-GB weights; record their PVC/S3/OCI URI and hash.

```bash
python scripts/run_eval.py --track prepared_diagnostics --config configs/eval.yaml
python scripts/run_eval.py --track tau_episodes --config configs/eval.yaml --limit 5 --trials 1
python scripts/run_eval.py --track retention --config configs/eval.yaml
```

Notebooks and CLI commands must call the same library. `08_compare_results.ipynb` must load actual MLflow results and compare task success, diagnostic accuracy, retention, latency, and failures. Do not insert fabricated performance values.

## 8. Repository Structure and Implementation Contracts

| Area | Required deliverables |
|---|---|
| Root | README.md, CLAUDE.md, pyproject.toml, Makefile, .env.example, .gitignore |
| Configurations | data-preparation.yaml, sdg.yaml, data-release.yaml, LoRA/OSFT profiles, rag.yaml, endpoints.example.yaml, eval.yaml |
| Environments | Locks/constraints and kernel instructions for preparation, LoRA, OSFT, backend, and τ evaluation |
| Notebooks | Eight learner notebooks and three authoring notebooks listed in Section 5.1 |
| `src/rhoai_model_training_lab` | config, schemas, data, sdg, training, deployment, rag, harness, api, tau_adapter, evaluation, tracking |
| Scripts | Source/SDG/validation/bundle/fetch/verify, train/export/upload/render, index/start/stop/smoke, run_eval/log_eval_results |
| Documentation | Architecture, reference audit, compatibility, data card, data release guide, evaluation protocol, troubleshooting, implementation status |
| Deployment | Model-serving manifests, backend manifests, Containerfile |
| Tests | Fixtures, split leakage, backend export parity/loss, tool replay, session isolation, official scorer adapter, resume/MLflow failures |

Put implementation logic in `src`; do not duplicate it across notebooks. Separate teacher, validator, user simulator, Base, LoRA, OSFT, embedding, MLflow, S3, and CA settings in `.env.example`. Identify which settings are required for learner training versus data authoring or evaluation. Do not commit private evaluation data, secrets, model weights, or executed notebook outputs.

## 9. Acceptance Criteria

| Gate | Required evidence |
|---|---|
| G0: Audit | Actual τ/SDG/Training Hub signatures and versions; verified tool discovery, reward, and split behavior |
| G1: Prepared data | Real bundle and hashes, quality report, training/evaluation isolation, LoRA/OSFT sample parity |
| G2: Training readiness | Both backend loaders/loss masks validated from the bundle without a teacher; GPU smoke tests when available |
| G3: Serving | Export/reload, manifests, live chat/tool-call checks or a specific blocker |
| G4: Backend | Persistent index, session isolation, real planning branches, official simulator tool integration, budget termination |
| G5: Evaluation | Official success/failure fixtures, live episodes, MLflow write/read or a specific blocker |
| G6: Reproducibility | Fresh-kernel learner path, authoring regeneration path, actual results and limitations |

Distinguish `data_ready`, `backend_load_validated`, `gpu_smoke_verified`, and `tau_live_verified`. Creating files does not establish successful GPU training. Run tokenizer/loss/schema validation without a GPU where possible.

When execution resources are unavailable, finish implementable code and procedures and mark the relevant gate blocked. Do not present fixtures or dry runs as live results. If no generation endpoint is available, do not label dummy data as a prepared lab dataset. Lead the completion report with the actual data readiness status.

## 10. Master Implementation Prompt for Opus

```text
You are an engineer implementing RHOAI 3.5 workflows, LLM post-training,
data engineering, and agent evaluation.
Read this entire rhoai-model-training-lab-opus-spec.md and complete an executable repository.
Read applicable repository instructions, preserve user changes, and resume from the recorded status.

The domain is fixed: τ-Knowledge banking_knowledge.
The highest-priority requirement is DATA-FIRST:
An operator uses sdg_hub to generate, validate, split, and export a prepared dataset bundle.
A learner downloads this bundle and runs simple LoRA/OSFT notebooks without a teacher API or SDG.

Required references:
https://github.com/sierra-research/tau2-bench
https://arxiv.org/abs/2603.04370
https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub
https://github.com/Red-Hat-AI-Innovation-Team/training_hub
https://github.com/cbtham/micro-financial-loan/tree/main
https://github.com/hyogrin/rhoai-custom-research-lab
https://github.com/hyogrin/rhoai-inference-design-planner/blob/main/scripts/run_eval.py

Inspect actual source, signatures, tasks, tools, rewards, and splits; pin real versions.
Do not assume upstream main equals the product-packaged version.
Implement in this order:
G0 audit → 1A prepared data → 1B training/deployment → 2 harness → 3 evaluation.

Mandatory rules:
- Use actual sdg_hub and training_hub APIs; isolate compatibility handling in thin adapters.
- Train LoRA and OSFT independently from the same base and canonical samples.
- Do not generate, repair, split, or perform complex data conversion in training notebooks.
- Validate that both backends load the data and apply assistant-only loss correctly.
- Validate tool trajectories using schemas and actual simulator replay.
- Never expose private evaluation fields, expected actions, or golden document lists to SDG or agents.
- Give all variants identical business tools and permitted current-account observations.
- Separate lab-derived offline diagnostics from official multi-turn task success.
- Reuse run_eval.py's MLflow patterns, not its 100-point text similarity rubric.
- Model evaluation must call actual serving endpoints.
- Log grading versions, simulator identity, tasks/trials, corpus, model, and configuration hashes.
- Never fabricate data, scores, URLs, or release readiness.

Deliver the package, prepared datasets, authoring and learner notebooks, scripts,
deployment manifests, backend, τ runner adapter, MLflow evaluation, and meaningful tests.
Keep this specification and coding instructions in English; provide Korean learner explanations
as specified in Section 1. Keep benchmark/training content in its original English.
Do not stop after producing a plan or scaffold.
Run feasible code/fixture checks and report resource blockers with precise commands.
Update docs/implementation-status.md after every stage.
```

## 11. Stage 1A Prompt: Prepare the Dataset in Advance

```text
Read the common specification and implement the prepared dataset release first.
Do not put SDG in the learner training notebooks.

1. Inspect the banking KB, public policies/tool schemas, official tasks/splits, and private fields.
2. Reserve official evaluation tasks and design SDG from the KB plus independent training scenarios.
   If an official training split exists, document its allowed use and contamination checks.
3. Implement sdg_hub YAML flows for QA, policy application, clarification, tool calls, and trajectories.
4. Implement grounding/condition checks, independent validation, actual simulator replay,
   deduplication, and split-leakage checks.
5. Export canonical train/validation samples into backend-specific JSONL files.
   Verify sample parity, chat templates, tool serialization, assistant-only masks, and lengths.
6. Package manifests, checksums, provenance, dataset cards, quality reports, and configurations.
7. Implement fetch/verify scripts and prepare a real small smoke bundle and a lab release when feasible.
8. Write the three authoring notebooks for source preparation, SDG, and validation/release.

Include only redistributable data and keep the private evaluation bundle separate.
Record review coverage; do not assume policy extraction or replay checks are infallible.
If no actual generation endpoint is available, report the blocker and do not pass G1 using dummy data.
Lead the completion report with data_ready and backend_load_validated status.
```

## 12. Stage 1B Prompt: Simple Training and Deployment

```text
Read the specification and prepared bundle manifest; implement the learner training path.
Create notebooks 00 preflight, 01 bundle loading, 03 LoRA, 04 OSFT, and 05 export/deployment.
Target 5–7 executable cells in each of 03/04: configuration, data validation/preview,
training, logs, and checkpoint verification.
Do not include source collection, teacher calls, SDG, splitting, or complex data conversion.

Start both algorithms from the same base revision and canonical dataset.
Verify and call actual training_hub APIs and expose the important steps for learning.
If the bundle is invalid, fail validation and link to the authoring path instead of modifying it.
Implement per-environment locks, GPU preflight, loss masking, checkpoint/resume, and MLflow logging.
Save LoRA adapters and merged models, and export OSFT artifacts to Hugging Face format.
Validate reloading. Adapt micro-financial-loan's export/S3/serving patterns to RHOAI 3.5.
Run a tool-call smoke test with the actual vLLM chat template and tool parser.
Without a GPU, finish code/loader checks and document concrete resource blockers.
Do not claim training success without evidence.
```

## 13. Stage 2 Prompt: RAG and Official Simulation Integration

```text
Read the specification and existing package; implement the RAG/planning backend.
Inspect graph/state/layers/API/observability in rhoai-custom-research-lab and reuse a small structure.
Connect FastAPI, LangGraph, a persistent vector index, and a τ simulation adapter.
Separate KB and training-example indexes and verify corpus/embedding/version fingerprints.

Provide /v1/qa for diagnostics and /v1/agent/step for the official agent runner.
Implement plan → retrieval/tool discovery → policy application → tool call → observation → verification.
Do not expose private user-simulator instructions, expected actions, or target reward states.
Connect business tools to the official simulation; never replace current observations with training values.
Preserve official tool discovery rather than exposing every tool schema upfront by default.
Implement session isolation, episode resets, bounded retries/budgets, and duplicate mutation prevention.
Provide simple_rag/agent_rag modes and a training-example retrieval ablation.
Create setup/index/start/stop/smoke scripts, notebook 06, Containerfile, and manifests.
Connect only to the simulation, not real banking APIs.
```

## 14. Stage 3 Prompt: Evaluation and MLflow

```text
Implement prepared_diagnostics, tau_episodes, and retention as defined in the specification.
Read the reference run_eval.py and reuse only its CLI/MLflow organization.
Do not use a FinQA numeric-program scorer or a 100-point golden-text rubric for this repository.

Preserve official τ runner/grader semantics while connecting live RHOAI model endpoints and backend APIs.
Test that private task fields never enter model requests.
Compare Base/LoRA/OSFT × no_knowledge/RAG with identical controllers, business tools, and simulators.
Run separate controlled ablations for planning and example retrieval.
Use official task success and pass^k; do not confuse pass^k with pass@k.
Label synthetic single-turn diagnostics as lab-derived metrics.
Validate the grader adapter using known-success and known-failure trajectories.
Include failures/timeouts in denominators and support task-level paired confidence intervals.
Create notebooks 07 and 08, run_eval/log_eval_results CLIs, and evaluation configuration.
Log data→training→evaluation lineage, grading revisions, simulator/configuration/model hashes,
rewards, diagnostics, latency/tokens/calls, failures, and redacted traces to MLflow.
Distinguish live persistence/retrieval checks from fixture checks and support offline re-upload.
```

## 15. Final Review Prompt

```text
Review the implementation against this specification and fix defects.
Verify the following:
- Can a learner train from the prepared bundle without a teacher API?
- Are real data bytes, manifests, checksums, and quality reports present?
- Do LoRA and OSFT use equivalent canonical samples and assistant loss targets?
- Are tool histories/serialization compatible with inference templates?
- Were synthetic trajectories validated in the actual simulation?
- Is KB adaptation distinguished from task leakage, with private evaluation fields isolated?
- Do all variants have identical account tools and observation permissions?
- Are unofficial diagnostics kept separate from official τ scores?
- Is simulator state isolated and reset for every task/trial?
- Are grading versions and pass^k semantics preserved?
- Do MLflow lineage, failure/offline states, and validation claims match actual evidence?
- Is the fresh-kernel learner notebook path short, clear, and sequentially executable?
Report data readiness first, followed by performed validation, unverified/blocked items,
and exact commands for remaining execution.
```
