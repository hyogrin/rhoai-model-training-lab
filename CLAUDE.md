# CLAUDE.md — Repository Instructions

## Project Overview

RHOAI 3.5 τ-Knowledge banking model training lab. Trains LoRA and OSFT
adapters for `Qwen/Qwen3-4B-Instruct-2507` on τ-Knowledge `banking_knowledge`
domain, with a RAG + planning harness and official τ evaluation.

## Language Policy

- Code identifiers, configuration keys, log keys: **English**
- Notebook explanations and learner documentation: **Korean**
- Benchmark and training content: **original English**
- This specification and coding prompts: **English**

## Architecture

```
src/rhoai_model_training_lab/
├── config/          # Centralized configuration loading
├── schemas/         # Pydantic data models
├── data/            # Data loading, validation, bundle management
├── sdg/             # Synthetic data generation (sdg_hub integration)
├── training/        # LoRA and OSFT training (training_hub integration)
├── deployment/      # Model export, S3 upload, serving manifests
├── rag/             # Vector index, BM25, retrieval
├── harness/         # LangGraph planning and tool execution
├── api/             # FastAPI endpoints
├── tau_adapter/     # τ-bench simulation adapter
├── evaluation/      # Diagnostics, episodes, retention evaluation
└── tracking/        # MLflow integration
```

## Key Principles

1. **DATA-FIRST**: Prepared dataset bundle before training notebooks
2. **No SDG in learner path**: Learner notebooks consume pre-built bundles
3. **Independent training**: LoRA and OSFT start from same base, same data
4. **Actual API usage**: Use real `sdg_hub` and `training_hub` APIs
5. **Evaluation integrity**: Never expose private eval fields to training/agent

## Development Commands

```bash
make install          # Install in development mode
make lint             # Run ruff
make test             # Run pytest
make test-unit        # Run unit tests only
make preflight        # Validate configs, schemas, bundle structure
```

## Environment Profiles

- `preparation`: SDG and data authoring (requires teacher endpoint)
- `lora`: LoRA fine-tuning (requires GPU)
- `osft`: OSFT fine-tuning (requires GPU)
- `backend`: RAG harness serving
- `evaluation`: τ episode evaluation (requires model endpoint + simulator)
