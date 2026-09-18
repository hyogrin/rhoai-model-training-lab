# Architecture Overview

## Component Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                     RHOAI Model Training Lab                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐                 │
│  │  config/  │   │ schemas/ │   │    cli.py    │                 │
│  │  Loading  │◄──│ Pydantic │◄──│  Click CLI   │                 │
│  │  & Env    │   │  Models  │   │              │                 │
│  └────┬─────┘   └────┬─────┘   └──────┬───────┘                 │
│       │              │                 │                          │
│  ┌────▼──────────────▼─────────────────▼──────────────────────┐  │
│  │                    Data Layer                               │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐    │  │
│  │  │   data/    │  │    sdg/      │  │  deployment/    │    │  │
│  │  │ BundleMgr  │  │ SDGPipeline  │  │ ModelExporter   │    │  │
│  │  │ Checksums  │  │ Validator    │  │ S3Uploader      │    │  │
│  │  │ Validation │  │ BundleBuilder│  │ ManifestRender  │    │  │
│  │  └────┬───────┘  └──────┬───────┘  └────────┬────────┘    │  │
│  └───────┼─────────────────┼────────────────────┼─────────────┘  │
│          │                 │                    │                 │
│  ┌───────▼─────────────────▼────────────────────▼─────────────┐  │
│  │                   Training Layer                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │  training/   │  │  training/   │  │   tracking/     │  │  │
│  │  │ LoRATrainer  │  │ OSFTTrainer  │  │ MLflowTracker   │  │  │
│  │  │ Preprocessor │  │ Decomposer   │  │ LocalResultStore│  │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Serving Layer                            │  │
│  │  ┌──────────┐  ┌────────────┐  ┌────────────────────────┐ │  │
│  │  │  rag/    │  │  harness/  │  │        api/            │ │  │
│  │  │ Embedding│  │ PlannerNode│  │ FastAPI (/healthz,     │ │  │
│  │  │ VectorIdx│  │ Retriever  │  │  /v1/agent/step,       │ │  │
│  │  │ BM25     │  │ PolicyNode │  │  /v1/qa)               │ │  │
│  │  │ Chunking │  │ ToolExec   │  │                        │ │  │
│  │  │ Retriever│  │ Verifier   │  │                        │ │  │
│  │  └──────────┘  │ Responder  │  └────────────────────────┘ │  │
│  │                └────────────┘                              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Evaluation Layer                           │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │  │
│  │  │ tau_adapter/  │  │ evaluation/   │  │ evaluation/  │  │  │
│  │  │ Simulation    │  │ PreparedDiag  │  │ ConfidenceCI │  │  │
│  │  │ AgentBridge   │  │ TauEpisodes   │  │ Orchestrator │  │  │
│  │  │ PrivateFilter │  │ Retention     │  │              │  │  │
│  │  └───────────────┘  └───────────────┘  └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

External Dependencies:
  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐
  │ Qwen3-4B     │  │ τ2-bench       │  │ MLflow Tracking      │
  │ (vLLM)       │  │ banking_knowl. │  │ Server               │
  └──────────────┘  └────────────────┘  └──────────────────────┘
  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐
  │ sdg_hub      │  │ training_hub   │  │ S3 / MinIO           │
  │ (teacher)    │  │ (LoRA/OSFT)    │  │ (model storage)      │
  └──────────────┘  └────────────────┘  └──────────────────────┘
```

## Data Flow

```
                    ┌─────────────────────┐
                    │   KB Documents       │
                    │ (banking_knowledge)  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   SDG Pipeline       │ ← preparation profile only
                    │  (Policy Extraction  │
                    │   → Teacher Model    │
                    │   → Validation)      │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Prepared Bundle     │
                    │  ├── canonical/      │
                    │  ├── training/       │
                    │  ├── kb/             │
                    │  ├── metadata/       │
                    │  └── manifest.json   │
                    └───┬─────────────┬───┘
                        │             │
              ┌─────────▼──┐    ┌─────▼─────────┐
              │ LoRA SFT   │    │  OSFT          │
              │ (peft +    │    │  (SVD decomp + │
              │  trainer)  │    │   selective    │
              │            │    │   unfreeze)    │
              └─────┬──────┘    └──────┬────────┘
                    │                  │
              ┌─────▼──────┐    ┌──────▼────────┐
              │ LoRA Merged│    │ OSFT Exported │
              │ Model      │    │ Model         │
              └─────┬──────┘    └──────┬────────┘
                    │                  │
                    └──────┬───────────┘
                           │
              ┌────────────▼────────────┐
              │    vLLM Serving          │
              │  (KServe / RHOAI 3.5)   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   RAG Harness Backend    │
              │  (ChromaDB + LangGraph)  │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
  ┌──────▼─────┐   ┌──────▼──────┐   ┌───────▼──────┐
  │ Track A    │   │ Track B     │   │ Track C      │
  │ Prepared   │   │ τ Episodes  │   │ Retention    │
  │ Diagnostics│   │ (official)  │   │ (ARC-Chall.) │
  └────────────┘   └─────────────┘   └──────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config/` | YAML loading, `${VAR}` expansion, profile validation |
| `schemas/` | Pydantic models for data, training, evaluation, API |
| `data/` | Bundle loading, checksum verification, JSONL parsing |
| `sdg/` | Policy extraction, teacher-model generation, validation, bundle building |
| `training/` | LoRA SFT, OSFT training, chat-template preprocessing, loss masking |
| `deployment/` | Model export, merge, S3 upload, KServe manifest rendering |
| `rag/` | Embedding, ChromaDB vector index, BM25, KB/example indexing, retrieval |
| `harness/` | LangGraph state-graph: planning, retrieval, policy, tool execution, verification |
| `api/` | FastAPI endpoints: `/healthz`, `/readyz`, `/v1/agent/step`, `/v1/qa` |
| `tau_adapter/` | τ-bench simulation bridge, private-field filtering, tool validation |
| `evaluation/` | Three evaluation tracks, pass^k computation, bootstrap CIs, orchestration |
| `tracking/` | MLflow logging, local JSONL persistence, fingerprint deduplication |
| `cli.py` | Click CLI: `preflight`, `train`, `evaluate`, `log-results`, `serve` |

## Key Design Decisions

1. **DATA-FIRST**: The prepared dataset bundle is built once (by the
   preparation profile) and consumed read-only by all downstream stages.
   Learner notebooks never invoke SDG.

2. **Independent Training**: Both LoRA and OSFT start from the same base
   checkpoint (`Qwen/Qwen3-4B-Instruct-2507`), share the same seed (42),
   and operate on identical canonical sample IDs.

3. **Evaluation Integrity**: Private evaluator fields (`expected_action`,
   `hidden_goal`, `target_reward`, etc.) are never exposed to the model
   or training data via `filter_private_fields()`.

4. **Graceful Degradation**: Heavy dependencies (`torch`, `transformers`,
   `peft`, `tau2`, `sdg_hub`, `training_hub`, `chromadb`) are imported
   behind `try/except` blocks so that lightweight operations (schema
   validation, config loading, testing) work without them.

5. **Local-First Persistence**: All evaluation and training results are
   saved locally via `LocalResultStore` before attempting MLflow logging.
   The `log-results` CLI command re-uploads pending records.
