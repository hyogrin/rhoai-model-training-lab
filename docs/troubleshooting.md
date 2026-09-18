# Troubleshooting

Common issues encountered when working with the RHOAI Model Training Lab
and their solutions.

## GPU / CUDA Issues

### GPU Out of Memory (OOM)

**Symptom:** `torch.cuda.OutOfMemoryError` or `CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   # configs/lora.yaml or configs/osft.yaml
   training_args:
     per_device_train_batch_size: 1   # reduce from 2
     gradient_accumulation_steps: 16  # increase to compensate
   ```

2. **Enable gradient checkpointing** (saves ~30% VRAM):
   ```yaml
   training_args:
     gradient_checkpointing: true
   ```

3. **Reduce sequence length:**
   ```yaml
   data:
     max_seq_length: 2048  # reduce from 4096
   ```

4. **For LoRA, reduce rank:**
   ```yaml
   lora:
     r: 8        # reduce from 16
     lora_alpha: 16  # reduce proportionally
   ```

5. **For OSFT, reduce unfreeze ratio:**
   ```yaml
   osft:
     unfreeze_rank_ratio: 0.15  # reduce from 0.25
   ```

6. **Check for leaked GPU memory:**
   ```bash
   nvidia-smi  # check if other processes are using VRAM
   ```

### No CUDA GPU Detected

**Symptom:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Verify CUDA driver: `nvidia-smi`
2. Verify PyTorch CUDA build: `python -c "import torch; print(torch.version.cuda)"`
3. Check GPU assignment in OpenShift: `oc describe pod <pod-name> | grep nvidia`
4. Ensure correct PyTorch version for your CUDA version

---

## Bundle Validation Failures

### manifest.json Not Found

**Symptom:** `FileNotFoundError: No manifest.json in bundle`

**Solutions:**
1. Verify bundle path: `ls data/prepared/tau-knowledge-v1/manifest.json`
2. Re-fetch the bundle: `python scripts/fetch_prepared_dataset.py`
3. Check PVC mount if on OpenShift: `oc exec <pod> -- ls /data/bundles/`

### Checksum Mismatch

**Symptom:** `Checksum mismatch for canonical/train.jsonl`

**Cause:** File was modified after bundle creation (corruption or tampering).

**Solutions:**
1. Re-download the bundle from the source
2. Verify network transfer completed: compare file sizes
3. If using S3: re-download with `--checksum-verify true`

### Sample Count Mismatch

**Symptom:** `Sample count mismatch in train: manifest says 200, file has 198`

**Cause:** Truncated file or incomplete download.

**Solutions:**
1. Re-download the bundle
2. Check for empty lines: `wc -l canonical/train.jsonl`
3. Validate JSON: `python -c "import json; [json.loads(l) for l in open('canonical/train.jsonl')]"`

### Missing Expected Files

**Symptom:** `Missing expected file: kb/documents.jsonl`

**Solutions:**
1. Ensure the complete bundle was downloaded
2. Check file permissions: `ls -la kb/`
3. Rebuild bundle if you have the source data

---

## MLflow Connection Issues

### Cannot Connect to MLflow

**Symptom:** `ConnectionError` or `MLflow connection failed`

**Solutions:**

1. **Check the URI:**
   ```bash
   echo $MLFLOW_TRACKING_URI
   curl -s $MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/list
   ```

2. **Run without MLflow** (results saved locally):
   ```bash
   rhoai-lab evaluate --no-mlflow
   rhoai-lab train --profile lora --no-mlflow
   ```

3. **Re-upload later:**
   ```bash
   rhoai-lab log-results --category all
   ```

4. **Check OpenShift Route:**
   ```bash
   oc get routes -n <namespace> | grep mlflow
   ```

### MLflow Artifact Too Large

**Symptom:** `Artifact is X MB (limit Y MB); skipping upload`

**Solutions:**
1. Model weights are never auto-uploaded (by design) — only metadata
2. Increase the limit if needed (not recommended for model weights):
   ```python
   tracker = MLflowTracker(artifact_max_size_mb=200)
   ```

---

## Model Endpoint Issues

### Authentication Failed (401/403)

**Symptom:** `RuntimeError: Authentication failed (401): check SERVING_TOKEN`

**Solutions:**
1. Verify the token:
   ```bash
   echo $SERVING_TOKEN
   curl -H "Authorization: Bearer $SERVING_TOKEN" \
     $BASE_SERVING_ENDPOINT/v1/models
   ```
2. Check token permissions in the serving configuration
3. For OpenShift: verify the Secret is mounted correctly

### Model Endpoint Timeout

**Symptom:** `httpx.TimeoutException` or `Model endpoint timeout (504)`

**Solutions:**
1. **Increase timeout:**
   ```yaml
   # configs/eval.yaml
   tracks:
     tau_episodes:
       budgets:
         max_wall_time_seconds: 600
   ```

2. **Check model health:**
   ```bash
   curl $BASE_SERVING_ENDPOINT/health
   curl $BASE_SERVING_ENDPOINT/v1/models
   ```

3. **Check GPU utilization** (model may be swapping):
   ```bash
   oc exec <serving-pod> -- nvidia-smi
   ```

4. **Reduce max_model_len** in the serving configuration:
   ```yaml
   args:
     - "--max-model-len"
     - "2048"
   ```

### Tool Calls Not Generated

**Symptom:** Model responds with text instead of tool calls

**Solutions:**
1. Verify `--tool-parser-plugin hermes` in the InferenceService manifest
2. Ensure `--enable-auto-tool-choice` is set
3. Check that tools are passed in the request payload
4. Test with the smoke test:
   ```python
   from rhoai_model_training_lab.deployment import DeploymentValidator
   validator = DeploymentValidator()
   result = validator.smoke_test_tool_call(endpoint_url, model_name=model_name)
   print(result)
   ```

---

## τ-bench Installation Issues

### τ2-bench Not Found

**Symptom:** `ImportError: No module named 'tau2'`

**Solutions:**
1. Install τ2-bench:
   ```bash
   pip install tau2-bench
   ```

2. Verify installation:
   ```python
   from rhoai_model_training_lab.tau_adapter import is_tau_available, get_tau_version
   print(is_tau_available(), get_tau_version())
   ```

3. Check Python version (requires ≥ 3.11):
   ```bash
   python --version
   ```

### τ Version Compatibility Warning

**Symptom:** `τ2-bench version X not in tested versions`

**Solutions:**
1. This is a warning, not an error — evaluation will proceed
2. Pin to a tested version:
   ```bash
   pip install tau2-bench==1.0.1
   ```
3. Run compatibility check:
   ```python
   from rhoai_model_training_lab.tau_adapter import check_tau_compatibility
   print(check_tau_compatibility())
   ```

---

## Backend / RAG Issues

### ChromaDB Import Error

**Symptom:** `ImportError: chromadb is required for the vector index`

**Solution:**
```bash
pip install -e ".[backend]"
```

### Sentence-Transformers Import Error

**Symptom:** `ImportError: sentence-transformers is required for local embedding`

**Solution:**
```bash
pip install -e ".[backend]"
```

### Empty Retrieval Results

**Symptom:** RAG returns no documents

**Solutions:**
1. Check if the index is built:
   ```python
   from rhoai_model_training_lab.rag import VectorIndex
   idx = VectorIndex()
   print(idx.collection_count("kb_documents"))
   ```
2. Rebuild the index from bundle KB documents
3. Check embedding dimension mismatch

---

## Common CLI Issues

### `rhoai-lab` Command Not Found

**Solution:**
```bash
pip install -e ".[dev]"
# or
python -m rhoai_model_training_lab.cli --help
```

### Config File Not Found

**Symptom:** `FileNotFoundError: Configuration file not found: configs/eval.yaml`

**Solution:**
```bash
ls configs/
# Ensure all required config files exist
# Copy from examples if needed
```
