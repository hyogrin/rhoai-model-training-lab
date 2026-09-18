# Data Release Guide

This guide covers how to prepare, build, validate, and publish a dataset
bundle for the RHOAI Model Training Lab.

## Overview

A **prepared dataset bundle** is a self-contained directory containing all
data needed for LoRA and OSFT training, RAG indexing, and evaluation. Learner
notebooks consume pre-built bundles — they never run SDG directly.

## Bundle Layout

```
tau-knowledge-v1/
├── manifest.json              # Bundle metadata and hashes
├── checksums.sha256           # SHA-256 of all files
├── DATASET_CARD.md            # Human-readable documentation
├── canonical/
│   ├── train.jsonl            # Canonical training samples
│   └── validation.jsonl       # Canonical validation samples
├── training/
│   ├── lora/
│   │   ├── train.jsonl        # Chat-template formatted for LoRA
│   │   └── validation.jsonl
│   └── osft/
│       ├── train.jsonl        # Chat-template formatted for OSFT
│       └── validation.jsonl
├── kb/
│   └── documents.jsonl        # Knowledge-base documents for RAG
├── metadata/
│   ├── provenance.jsonl       # Per-sample provenance records
│   └── splits.json            # Split info with family partitions
└── reports/
    ├── quality.json           # Quality assessment report
    └── quality.md             # Human-readable quality summary
```

## Step-by-Step: Building a Bundle

### 1. Prerequisites

- **preparation** environment profile installed
- Teacher model endpoint configured (for SDG)
- KB documents available in `data/source/`

```bash
pip install -e ".[sdg]"
cp .env.example .env
# Edit .env with teacher endpoint and API key
```

### 2. Configure SDG

Edit `configs/sdg.yaml` for target counts:

```yaml
generation:
  target_counts:
    lab:
      policy_qa: 80
      tool_selection: 40
      trajectory: 50
      policy_application: 20
      clarification: 10
      total: 200
```

### 3. Configure Data Release

Edit `configs/data-release.yaml`:

```yaml
bundle:
  name: tau-knowledge
  version: v1
  base_path: data/prepared/tau-knowledge-v1

model_profile:
  model_id: Qwen/Qwen3-4B-Instruct-2507
  model_revision: main
  tokenizer_id: Qwen/Qwen3-4B-Instruct-2507
```

### 4. Run the SDG Pipeline

```bash
python scripts/run_sdg.py --profile lab
```

This will:
1. Extract policies from KB documents
2. Generate synthetic samples via the teacher model
3. Validate each sample (schema, grounding, deduplication, contamination)
4. Checkpoint progress periodically

### 5. Build the Bundle

```bash
python scripts/build_bundle.py
```

This assembles the full bundle directory from accepted samples.

### 6. Validate the Bundle

```bash
rhoai-lab preflight
# or directly:
python -c "
from rhoai_model_training_lab.data import validate_prepared_dataset
result = validate_prepared_dataset('data/prepared/tau-knowledge-v1')
print('Valid:', result['valid'])
for e in result['errors']:
    print('  ERROR:', e)
for w in result['warnings']:
    print('  WARN:', w)
"
```

### 7. Publish the Bundle

#### To S3/MinIO:

```bash
python scripts/upload_bundle.py \
  --source data/prepared/tau-knowledge-v1 \
  --bucket rhoai-bundles \
  --prefix prepared/
```

#### To PVC (on OpenShift):

```bash
oc cp data/prepared/tau-knowledge-v1 \
  pod-name:/data/bundles/tau-knowledge-v1
```

## Validation Checks

The `validate_prepared_dataset()` function performs:

| Check | Description |
|-------|-------------|
| Manifest | `manifest.json` exists and is valid |
| Expected files | All required files present (canonical, kb, metadata, checksums) |
| Checksums | All files match their SHA-256 hashes |
| Sample counts | Canonical JSONL line counts match manifest |
| Split consistency | Split metadata matches manifest counts |

Optional file warnings (non-blocking):
- `reports/quality.json`
- `DATASET_CARD.md`
- `training/lora/*.jsonl`
- `training/osft/*.jsonl`

## Split Isolation Rules

The bundle builder enforces strict split isolation:

1. **Scenario family isolation**: All samples sharing a `scenario_family`
   are assigned to the same split (train OR validation, never both)
2. **No ID leakage**: No `sample_id` appears in both train and validation
3. **Holdout disjointness**: Reserved holdout IDs are disjoint from both
   training and validation sets
4. **Contamination check**: Validation and holdout IDs must not appear in
   the training set

## Updating an Existing Bundle

To create a new version:

1. Increment the version in `configs/data-release.yaml`
2. Re-run the SDG pipeline (it can resume from checkpoint)
3. Rebuild the bundle
4. Validate checksums and sample counts
5. Publish with the new version tag

Bundle versions are immutable once published. Never modify a published bundle
in place — always create a new version.
