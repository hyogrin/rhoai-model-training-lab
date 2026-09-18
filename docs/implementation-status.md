# Implementation Status

> Last updated: 2026-09-18  
> Specification: `rhoai-model-training-lab-opus-spec.md` (Revision 3)

---

## Summary

| Metric | Count |
|---|---|
| Total source files | 86 |
| Total lines (all source) | ~24,100 |
| Python source modules | 14,756 lines |
| Scripts (Python + Bash) | 4,035 lines |
| Tests | 1,962 lines |
| Notebooks | 11 (75 cells total) |
| Config files | 8 YAML |
| Deployment manifests | 6 |
| Documentation files | 6 |

---

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| **G0: Audit** | ⚠️ `structure_ready` | Reference audit document created (`docs/reference-audit.md`). Actual τ-bench, sdg_hub, training_hub API signatures must be verified against pinned versions at runtime. SHAs not yet pinned — requires installation. |
| **G1: Prepared data** | ⚠️ `code_ready` / `data_not_ready` | Full SDG pipeline, validation, bundle builder implemented. Authoring notebooks and scripts complete. **No actual data generated** — requires teacher endpoint (`SDG_TEACHER_ENDPOINT`). Smoke bundle not yet produced. |
| **G2: Training readiness** | ⚠️ `code_ready` / `gpu_not_verified` | LoRA and OSFT trainers implemented with `training_hub` adapter + native fallback. Bundle validation, loss masking, chat template checks implemented. **No GPU smoke test executed** — requires GPU Workbench. |
| **G3: Serving** | ⚠️ `manifests_ready` / `not_deployed` | Model export (LoRA merge, OSFT HF export), S3 upload, InferenceService manifests rendered. vLLM tool-parser config included. **Not deployed** — requires cluster access. |
| **G4: Backend** | ⚠️ `code_ready` / `not_live_tested` | FastAPI + LangGraph harness, ChromaDB/BM25 retrieval, τ simulation adapter, session isolation, budget enforcement all implemented. 155 tests pass. **No live integration test** — requires model endpoint + τ-bench installation. |
| **G5: Evaluation** | ⚠️ `code_ready` / `not_live_tested` | Three tracks (prepared_diagnostics, tau_episodes, retention) implemented. pass^k semantics, confidence intervals, MLflow logging all present. **No live episodes executed** — requires model endpoint + τ-bench + user simulator. |
| **G6: Reproducibility** | ⚠️ `structure_verified` / `end_to_end_not_verified` | Fresh-kernel learner path designed (notebooks 00→01→03/04→05→06→07→08). Authoring path notebooks complete. **Full end-to-end run not yet performed.** |

---

## Deliverables Checklist

### Root Files
- [x] `README.md` — Project overview, quickstart, architecture
- [x] `CLAUDE.md` — Repository instructions for AI agents
- [x] `pyproject.toml` — Package definition with optional dependency groups
- [x] `Makefile` — Development commands
- [x] `.env.example` — All environment variables with descriptions
- [x] `.gitignore` — Python/Jupyter/IDE ignores

### Configuration (`configs/`)
- [x] `data-preparation.yaml` — τ-bench source preparation
- [x] `sdg.yaml` — Synthetic data generation pipeline
- [x] `data-release.yaml` — Bundle build and fetch
- [x] `lora.yaml` — LoRA fine-tuning profile
- [x] `osft.yaml` — OSFT fine-tuning profile
- [x] `rag.yaml` — RAG harness and retrieval
- [x] `endpoints.example.yaml` — Model serving endpoints
- [x] `eval.yaml` — Evaluation configuration (3 tracks, variants, ablations)

### Source Package (`src/rhoai_model_training_lab/`)
- [x] `config/` — YAML loading, env var expansion (114 lines)
- [x] `schemas/` — Pydantic models: data, training, api, evaluation (494 lines)
- [x] `data/` — Bundle manager, fetch, validate, checksums (589 lines)
- [x] `sdg/` — SDG pipeline, policy extractor, validator, bundle builder (943 lines)
- [x] `training/` — LoRA/OSFT trainers, data preprocessor, loss masking (1,027 lines)
- [x] `deployment/` — Model export, S3 upload, manifest rendering, smoke tests (808 lines)
- [x] `rag/` — Embedding, ChromaDB, BM25, KB/example indexers, retriever (825 lines)
- [x] `harness/` — LangGraph state graph, planner/retriever/policy/tool/verifier nodes (758 lines)
- [x] `api/` — FastAPI app: /healthz, /readyz, /v1/agent/step, /v1/qa (481 lines)
- [x] `tau_adapter/` — τ simulation adapter, agent bridge, private field filtering (626 lines)
- [x] `evaluation/` — 3 track evaluators, pass^k, confidence intervals, orchestrator (1,588 lines)
- [x] `tracking/` — MLflow tracker, local result store, fingerprint dedup (750 lines)
- [x] `cli.py` — Click CLI: preflight, train, evaluate, log-results, serve (628 lines)

### Scripts (`scripts/`)
- [x] `setup_env.sh` — Profile-based environment setup (263 lines)
- [x] `fetch_prepared_dataset.py` — Local/S3/PVC bundle fetch (306 lines)
- [x] `validate_prepared_dataset.py` — Bundle integrity validation (369 lines)
- [x] `prepare_tau_sources.py` — τ-bench source extraction (344 lines)
- [x] `generate_synthetic.py` — SDG Hub pipeline execution (371 lines)
- [x] `validate_synthetic.py` — Independent data validation (385 lines)
- [x] `build_prepared_bundle.py` — Bundle assembly (495 lines)
- [x] `build_index.sh` — ChromaDB + BM25 index building (277 lines)
- [x] `start_backend.sh` — FastAPI server start (146 lines)
- [x] `stop_backend.sh` — Server shutdown (119 lines)
- [x] `smoke_backend.sh` — Health and endpoint checks (178 lines)
- [x] `run_eval.py` — 3-track evaluation runner (512 lines)
- [x] `log_eval_results.py` — MLflow re-upload (270 lines)

### Notebooks
| # | Notebook | Audience | Cells | Status |
|---|---|---|---|---|
| 00 | `preflight` | Learner | 7 | ✅ Code ready |
| 01 | `load_prepared_dataset` | Learner | 7 | ✅ Code ready |
| 03 | `lora_finetuning` | Learner | 7 | ✅ Code ready (no SDG/teacher) |
| 04 | `osft_finetuning` | Learner | 7 | ✅ Code ready (no SDG/teacher) |
| 05 | `export_and_deploy` | Learner | 7 | ✅ Code ready |
| 06 | `rag_harness` | Learner | 7 | ✅ Code ready |
| 07 | `evaluate` | Learner | 6 | ✅ Code ready |
| 08 | `compare_results` | Learner | 7 | ✅ Code ready |
| DP-01 | `prepare_tau_sources` | Author | 6 | ✅ Code ready |
| DP-02 | `generate_synthetic` | Author | 7 | ✅ Code ready |
| DP-03 | `validate_and_release` | Author | 7 | ✅ Code ready |

### Tests
- [x] `tests/conftest.py` — Shared fixtures (222 lines)
- [x] `tests/unit/test_schemas.py` — Pydantic model tests (332 lines)
- [x] `tests/unit/test_config.py` — Config loading tests (120 lines)
- [x] `tests/unit/test_data.py` — Bundle/data tests (201 lines)
- [x] `tests/unit/test_split_leakage.py` — Split isolation tests (123 lines)
- [x] `tests/unit/test_training.py` — Training module tests (143 lines)
- [x] `tests/unit/test_tau_adapter.py` — τ adapter tests (265 lines)
- [x] `tests/unit/test_evaluation.py` — Evaluation logic tests (228 lines)
- [x] `tests/unit/test_tracking.py` — MLflow tracking tests (192 lines)
- [x] `tests/integration/test_api.py` — FastAPI endpoint tests (136 lines)
- [x] `tests/fixtures/` — Known success/failure trajectory fixtures

### Documentation (`docs/`)
- [x] `architecture.md` — Component diagram, data flow, module responsibilities
- [x] `reference-audit.md` — τ-bench, sdg_hub, training_hub, references audit
- [x] `evaluation-protocol.md` — 3 tracks, pass^k semantics, CI methodology
- [x] `data-release-guide.md` — Operator bundle preparation guide
- [x] `troubleshooting.md` — Common issues and solutions
- [x] `implementation-status.md` — This file

### Deployment (`deployment/`)
- [x] `Containerfile` — Multi-stage backend build (arbitrary UID, non-root)
- [x] `manifests/inference-service-base.yaml` — Base model InferenceService
- [x] `manifests/inference-service-lora.yaml` — LoRA-merged InferenceService
- [x] `manifests/inference-service-osft.yaml` — OSFT InferenceService
- [x] `manifests/backend-deployment.yaml` — Backend Deployment with PVC/probes
- [x] `manifests/backend-service.yaml` — Backend Service
- [x] `manifests/backend-route.yaml` — OpenShift Route

---

## Spec Compliance Summary

### Mandatory Rules (Section 10)

| Rule | Status |
|---|---|
| Use actual `sdg_hub` and `training_hub` APIs via thin adapters | ✅ Implemented with try/except fallback |
| Train LoRA and OSFT independently from same base and canonical samples | ✅ Both use same `model_id`, `seed`, canonical IDs |
| No SDG/teacher/repair in training notebooks | ✅ Notebooks 03/04 consume bundle only |
| Validate both backends load data with assistant-only loss | ✅ `DataPreprocessor` + `validate_bundle_for_training()` |
| Validate tool trajectories via schema + simulation replay | ✅ `SyntheticValidator.validate_tool_call()` + simulation replay |
| Never expose private eval fields to SDG or agents | ✅ `filter_private_fields()` in tau_adapter |
| All variants get identical business tools and observations | ✅ Harness uses same tool set per session |
| Separate lab diagnostics from official τ scores | ✅ `prepared_diagnostics` labeled as "τ-Knowledge-derived lab diagnostics" |
| Reuse `run_eval.py` MLflow patterns, not text similarity rubric | ✅ Own evaluation with official τ grading |
| Model evaluation calls actual serving endpoints | ✅ `TauEpisodeRunner` calls live endpoints |
| Log grading versions, simulator identity, hashes | ✅ `EvalResult` captures all lineage |
| Never fabricate data, scores, URLs, or release readiness | ✅ All gates report actual status |

### Key Architectural Decisions

| Decision | Implementation |
|---|---|
| τ-Knowledge `banking_knowledge` domain | Hardcoded in configs, tau_adapter |
| Base model: `Qwen/Qwen3-4B-Instruct-2507` | Pinned in all configs |
| LoRA r=16/α=32, OSFT ratio=0.25 | In configs as initial candidates |
| FastAPI + LangGraph + ChromaDB | `api/`, `harness/`, `rag/` modules |
| simple_rag / agent_rag modes | Harness graph supports both |
| pass^k (not pass@k) semantics | `TauEpisodeRunner._compute_pass_k()` |
| MLflow 3 experiments with lineage | `MLflowTracker` with run ID linking |

---

## Blocked Items & Required Resources

To advance from `code_ready` to `live_verified`, the following resources are needed:

### For G1 (Prepared Data)
```bash
# Set teacher endpoint in .env
SDG_TEACHER_ENDPOINT=https://...
SDG_TEACHER_API_KEY=...
SDG_TEACHER_MODEL=...

# Then run:
python scripts/prepare_tau_sources.py --config configs/data-preparation.yaml
python scripts/generate_synthetic.py --config configs/sdg.yaml --profile smoke
python scripts/validate_synthetic.py --config configs/data-preparation.yaml
python scripts/build_prepared_bundle.py --config configs/data-release.yaml
```

### For G2 (Training)
```bash
# Requires: GPU Workbench + prepared bundle
bash scripts/setup_env.sh --profile lora
python scripts/fetch_prepared_dataset.py --release configs/data-release.yaml
# Run notebooks 03_lora_finetuning.ipynb and 04_osft_finetuning.ipynb
```

### For G3 (Serving)
```bash
# Requires: RHOAI 3.5 cluster + trained models on S3
oc apply -f deployment/manifests/inference-service-base.yaml
oc apply -f deployment/manifests/inference-service-lora.yaml
oc apply -f deployment/manifests/inference-service-osft.yaml
```

### For G4 (Backend)
```bash
# Requires: model endpoint + prepared bundle
bash scripts/setup_env.sh --profile backend
bash scripts/build_index.sh --bundle data/prepared/tau-knowledge-v1
bash scripts/start_backend.sh --host 127.0.0.1 --port 8000
bash scripts/smoke_backend.sh
```

### For G5 (Evaluation)
```bash
# Requires: model endpoints + τ-bench + user simulator
pip install tau2-bench  # or from pinned source
python scripts/run_eval.py --track tau_episodes --config configs/eval.yaml --limit 5 --trials 1
python scripts/run_eval.py --track prepared_diagnostics --config configs/eval.yaml
python scripts/run_eval.py --track retention --config configs/eval.yaml
```

### For G6 (Reproducibility)
- Run full learner path: notebooks 00 → 01 → 03 → 04 → 05 → 06 → 07 → 08
- Run full authoring path: DP-01 → DP-02 → DP-03
- Verify fresh-kernel sequential execution

---

## Version Pins (To Be Confirmed at Runtime)

| Component | Target Version | Pinned | Notes |
|---|---|---|---|
| τ-bench | v1.0.1 | ❌ | Need commit SHA after installation |
| sdg_hub | latest | ❌ | Verify API signatures |
| training_hub | latest | ❌ | Verify lora_sft/osft APIs |
| Qwen3-4B-Instruct-2507 | main | ❌ | Pin revision after compatibility test |
| vLLM runtime | RHOAI 3.5 bundled | ❌ | Check runtime image digest |
| Python | ≥ 3.11 | ✅ | In pyproject.toml |
