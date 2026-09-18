# RHOAI Model Training Lab

> **RHOAI 3.5 τ-Knowledge banking domain model training lab**
>
> Fine-tune LoRA and OSFT adapters on `Qwen/Qwen3-4B-Instruct-2507`,
> serve them through a RAG + planning harness, and evaluate with
> official τ-Knowledge episodes — all on Red Hat OpenShift AI.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  RHOAI Model Training Lab                     │
├──────────┬──────────┬──────────┬──────────┬──────────────────┤
│ config/  │ schemas/ │  data/   │  sdg/    │  deployment/     │
│ Loading  │ Pydantic │ Bundles  │ Pipeline │  Export & Serve  │
├──────────┴──────────┼──────────┼──────────┼──────────────────┤
│    training/        │ rag/     │ harness/ │  api/            │
│  LoRA + OSFT SFT    │ Embed +  │ LangGraph│  FastAPI         │
│  Preprocessing      │ Retrieve │ Planning │  /v1/agent/step  │
├─────────────────────┼──────────┴──────────┼──────────────────┤
│    tau_adapter/      │   evaluation/       │  tracking/       │
│  τ-bench Bridge      │  3 Tracks + CI      │  MLflow + Local  │
└─────────────────────┴─────────────────────┴──────────────────┘
```

## OpenShift AI Workbench Environment

All notebooks run inside **RHOAI Workbench** instances. Different stages
have different resource requirements.

### Workbench Requirements by Stage

| Stage | Notebooks | Workbench Type | Why |
|-------|-----------|----------------|-----|
| Preflight & data loading | `00`, `01` | CPU (Small) | Config validation, bundle fetch |
| Data preparation (operator) | `data_preparation/*` | CPU (Small) | SDG teacher API calls, no local compute |
| **LoRA / OSFT training** | **`03`, `04`** | **GPU (A10G 24 GB recommended)** | **Model training requires CUDA** |
| Export & deploy | `05` | CPU or GPU | Model merge can use CPU (slow) or GPU |
| RAG harness | `06` | CPU (Small) | Embedding + ChromaDB indexing |
| Evaluation & comparison | `07`, `08` | CPU (Small) | Calls deployed model endpoints |

### GPU Node Strategy

GPU nodes are expensive. Use an on-demand scaling approach:

```
MachineSet (e.g. g5.2xlarge — 1× A10G 24 GB)
├── replicas: 0   ← default (no cost)
└── replicas: 1   ← scale up only for training notebooks 03–04
```

**Before the lab**: Scale the GPU MachineSet to `replicas: 1` and wait
for the node to become `Ready`. Then create or start a GPU-enabled
Workbench from the RHOAI Dashboard.

**After training**: Stop the GPU Workbench, scale the MachineSet back to
`replicas: 0`, and continue with CPU notebooks for evaluation.

> **Tip for instructors**: For a classroom setting, scale up the GPU node
> before the session starts. Learners only need the GPU Workbench for
> notebooks 03 and 04 (~1–3 hours). All other notebooks run on a Small
> CPU Workbench.

### Recommended GPU Instances

| Instance (AWS) | GPU | VRAM | Qwen3-4B LoRA | Qwen3-4B OSFT |
|----------------|-----|------|---------------|---------------|
| **g5.2xlarge** | A10G ×1 | 24 GB | ✅ batch 2 | ✅ batch 1 |
| g5.4xlarge | A10G ×1 | 24 GB | ✅ batch 4 | ✅ batch 1–2 |
| p3.2xlarge | V100 ×1 | 16 GB | ⚠️ tight | ⚠️ tight |

The 4B parameter model fits comfortably on a single **A10G (24 GB)** for
both LoRA and OSFT. An A100 is not necessary.

## Quick Start (Learner Guide)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd rhoai-model-training-lab

# Install dependencies (choose the appropriate profile)
pip install -e ".[lora]"     # for LoRA training
pip install -e ".[osft]"     # for OSFT training

# Configure environment variables
cp .env.example .env
# Edit .env to set endpoint URLs and tokens
```

### 2. Preflight Check

```bash
rhoai-lab preflight
```

Automatically verifies GPU status, environment variables, data paths,
MLflow connectivity, and model profile.

### 3. Fetch the Prepared Data Bundle

Learners use a pre-built data bundle — SDG is not executed directly:

```bash
python scripts/fetch_prepared_dataset.py
```

### 4. Training

#### LoRA Fine-Tuning

```bash
# In a notebook (recommended):
# Open notebooks_ko/03_lora_finetuning.ipynb and follow step by step

# Via CLI:
rhoai-lab train --profile lora
```

#### OSFT Fine-Tuning

```bash
# Open notebooks_ko/04_osft_finetuning.ipynb

# Via CLI:
rhoai-lab train --profile osft
```

### 5. Evaluation

```bash
# All three tracks:
rhoai-lab evaluate --track all

# Individual tracks:
rhoai-lab evaluate --track prepared_diagnostics
rhoai-lab evaluate --track tau_episodes --trials 3
rhoai-lab evaluate --track retention

# With explicit endpoint mapping:
rhoai-lab evaluate --track all \
  --endpoint base https://base-model.apps.cluster.com/v1 \
  --endpoint lora https://lora-model.apps.cluster.com/v1 \
  --endpoint osft https://osft-model.apps.cluster.com/v1
```

### 6. View Results

```bash
# MLflow UI:
mlflow ui --port 5000

# Re-upload local results to MLflow:
rhoai-lab log-results --category all
```

## Authoring Path (Operator / Instructor)

To build a data bundle from scratch, use the **preparation** profile:

```bash
pip install -e ".[sdg]"

# Prepare τ sources
python scripts/prepare_tau_sources.py --config configs/data-preparation.yaml

# Generate synthetic data
python scripts/generate_synthetic.py --config configs/sdg.yaml

# Validate and build the bundle
python scripts/validate_synthetic.py --config configs/data-preparation.yaml
python scripts/build_prepared_bundle.py --config configs/data-release.yaml
```

See [docs/data-release-guide.md](docs/data-release-guide.md) for details.

## Evaluation Framework

| Track | Description | Metrics |
|-------|-------------|---------|
| **A: Prepared Diagnostics** | Holdout-based offline evaluation | Answer accuracy, tool accuracy, grounding score |
| **B: τ Episodes** | Official τ-Knowledge end-to-end episodes | pass^k (consistent success), action recall |
| **C: Retention** | ARC-Challenge capability preservation | retention_delta_pp (target ≥ −2.0 pp) |

**pass^k** (not pass@k): the model must succeed on *every* trial for a
given task. Consistent accuracy matters for banking agents.

See [docs/evaluation-protocol.md](docs/evaluation-protocol.md) for the
full protocol.

## Directory Structure

| Path | Description |
|------|-------------|
| `src/rhoai_model_training_lab/` | Main Python package |
| `src/.../config/` | YAML config loading with `${VAR}` expansion |
| `src/.../schemas/` | Pydantic data models |
| `src/.../data/` | Bundle management, checksum verification |
| `src/.../sdg/` | Synthetic data generation (preparation profile only) |
| `src/.../training/` | LoRA / OSFT training and preprocessing |
| `src/.../deployment/` | Model export, S3 upload, serving manifests |
| `src/.../rag/` | Vector index, BM25, retrieval |
| `src/.../harness/` | LangGraph planning and tool execution |
| `src/.../api/` | FastAPI backend endpoints |
| `src/.../tau_adapter/` | τ-bench simulation adapter |
| `src/.../evaluation/` | Three-track evaluation with bootstrap CI |
| `src/.../tracking/` | MLflow integration and local result store |
| `src/.../cli.py` | Click CLI entry point |
| `configs/` | YAML configuration files |
| `notebooks_ko/` | Jupyter lab notebooks (Korean explanations) |
| `notebooks_en/` | Jupyter lab notebooks (English explanations) |
| `scripts/` | Utility scripts |
| `tests/` | pytest test suite |
| `deployment/` | Containerfile and Kubernetes manifests |
| `docs/` | Architecture, protocol, and troubleshooting docs |

## Install Profiles

| Profile | Command | Purpose |
|---------|---------|---------|
| `dev` | `pip install -e ".[dev]"` | Testing and linting |
| `sdg` | `pip install -e ".[sdg]"` | SDG data authoring |
| `lora` | `pip install -e ".[lora]"` | LoRA training (GPU required) |
| `osft` | `pip install -e ".[osft]"` | OSFT training (GPU required) |
| `backend` | `pip install -e ".[backend]"` | RAG harness serving |
| `eval` | `pip install -e ".[eval]"` | Evaluation and MLflow |
| `all` | `pip install -e ".[all]"` | All dependencies |

## Development

```bash
# Development setup
pip install -e ".[dev]"

# Lint
make lint

# Tests
make test

# Unit tests only
make test-unit

# Preflight checks
make preflight
```

## Key Design Principles

1. **DATA-FIRST**: Prepared dataset bundle must exist before training notebooks
2. **No SDG in learner path**: Learners consume pre-built bundles
3. **Independent training**: LoRA and OSFT start from the same base model, seed, and data
4. **Real API usage**: Actual `sdg_hub` and `training_hub` APIs — no mocks
5. **Evaluation integrity**: Private evaluation fields are never exposed to training or agents

## Language Policy

| Content | Language |
|---------|----------|
| Code identifiers, config keys, log keys | English |
| Notebook explanations and learner docs | Korean (notebooks_ko) / English (notebooks_en) |
| Benchmark and training content | Original English |
| This README and specifications | English |

## Contributing

1. Create a branch with a `feature/` prefix
2. Write tests for all new code
3. Ensure `make preflight` passes
4. Run `make lint && make test` before submitting a PR
5. Follow the language policy above

## License

Apache-2.0
