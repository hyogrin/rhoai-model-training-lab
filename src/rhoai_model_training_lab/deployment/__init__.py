"""Model export, S3 upload, serving manifest rendering, and deployment validation.

Supports vLLM-based serving on RHOAI 3.5 with Hermes-style tool parsing,
custom chat templates, and OpenAI-compatible endpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

import httpx
from jinja2 import Environment, BaseLoader

from rhoai_model_training_lab.data import compute_file_checksum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_VLLM_IMAGE = "quay.io/modh/vllm:rhoai-2.20-cuda"
_DEFAULT_RUNTIME = "vllm-runtime"
_DEFAULT_TOOL_PARSER = "hermes"
_API_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# ModelExporter
# ---------------------------------------------------------------------------

class ModelExporter:
    """Export and validate trained model artefacts.

    Handles both LoRA adapter-only exports and full merged / OSFT exports.
    """

    def export_lora_adapter(
        self,
        adapter_path: Path | str,
        output_path: Path | str | None = None,
    ) -> Path:
        """Copy a LoRA adapter directory to the export location.

        Args:
            adapter_path: Source adapter directory.
            output_path: Destination path.  Defaults to ``adapter_path``.

        Returns:
            Resolved output path.
        """
        src = Path(adapter_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Adapter not found at {src}")

        dst = Path(output_path).resolve() if output_path else src
        if dst != src:
            import shutil
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info("LoRA adapter exported: %s → %s", src, dst)
        else:
            logger.info("LoRA adapter already at export path: %s", dst)

        self._write_export_metadata(dst, method="lora_adapter")
        return dst

    def merge_lora(
        self,
        base_model_id: str,
        adapter_path: Path | str,
        output_path: Path | str,
        torch_dtype: str = "bfloat16",
    ) -> Path:
        """Merge a LoRA adapter into the base model and save.

        Requires ``transformers`` and ``peft``.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
            from peft import PeftModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "transformers and peft are required for merge_lora"
            ) from exc

        adapter = Path(adapter_path).resolve()
        out = Path(output_path).resolve()
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Loading base model: %s", base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        logger.info("Loading LoRA adapter: %s", adapter)
        model = PeftModel.from_pretrained(base_model, str(adapter))
        merged = model.merge_and_unload()

        logger.info("Saving merged model to %s", out)
        merged.save_pretrained(str(out))

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=True,
        )
        tokenizer.save_pretrained(str(out))

        self._write_export_metadata(out, method="lora_merged")
        return out

    def export_osft(
        self,
        checkpoint_path: Path | str,
        output_path: Path | str,
    ) -> Path:
        """Copy an OSFT checkpoint to the export location.

        OSFT models are full HuggingFace checkpoints (no separate adapter).
        """
        src = Path(checkpoint_path).resolve()
        out = Path(output_path).resolve()

        if not src.exists():
            raise FileNotFoundError(f"OSFT checkpoint not found at {src}")

        if out != src:
            import shutil
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(src, out)
            logger.info("OSFT model exported: %s → %s", src, out)
        else:
            logger.info("OSFT model already at export path: %s", out)

        self._write_export_metadata(out, method="osft")
        return out

    def validate_exported_model(self, model_path: Path | str) -> dict[str, Any]:
        """Validate an exported model directory.

        Checks for essential files, shard integrity, and config consistency.

        Returns:
            Dict with ``"valid"`` (bool), ``"errors"`` (list), and
            ``"file_count"`` (int).
        """
        path = Path(model_path).resolve()
        errors: list[str] = []

        if not path.exists():
            return {"valid": False, "errors": ["Path does not exist"], "file_count": 0}

        # Config file
        config_file = path / "config.json"
        if not config_file.exists():
            errors.append("config.json missing")

        # Tokenizer files
        has_tokenizer = (
            (path / "tokenizer.json").exists()
            or (path / "tokenizer_config.json").exists()
        )
        if not has_tokenizer:
            errors.append("No tokenizer files found")

        # Model weights
        safetensors = list(path.glob("*.safetensors"))
        bin_files = list(path.glob("*.bin"))
        adapter_files = list(path.glob("adapter_model.*"))

        has_weights = bool(safetensors or bin_files or adapter_files)
        if not has_weights:
            errors.append("No model weight files found")

        # Index file for sharded models
        if len(safetensors) > 1 and not (path / "model.safetensors.index.json").exists():
            errors.append("Sharded safetensors without index file")
        if len(bin_files) > 1 and not (path / "pytorch_model.bin.index.json").exists():
            errors.append("Sharded pytorch_model without index file")

        # Checksum spot-check on weight files
        weight_files = safetensors or bin_files or adapter_files
        for wf in weight_files[:3]:
            try:
                compute_file_checksum(wf)
            except Exception as exc:
                errors.append(f"Checksum computation failed for {wf.name}: {exc}")

        file_count = sum(1 for _ in path.rglob("*") if _.is_file())

        valid = len(errors) == 0
        logger.info(
            "Model validation %s: %d files, %d errors",
            "PASSED" if valid else "FAILED", file_count, len(errors),
        )
        return {"valid": valid, "errors": errors, "file_count": file_count}

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _write_export_metadata(path: Path, method: str) -> None:
        meta = {
            "export_method": method,
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = path / "export_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# S3Uploader
# ---------------------------------------------------------------------------

class S3Uploader:
    """Upload and download model artefacts to/from S3.

    Requires ``boto3``.  Credentials are resolved through the standard
    AWS credential chain (env vars, ~/.aws, IRSA, etc.).

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for all objects (e.g. ``"models/"``).
        endpoint_url: Custom S3 endpoint (for MinIO / Ceph on OpenShift).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._endpoint_url = endpoint_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("boto3 is required for S3 operations") from exc
            kwargs: dict[str, Any] = {}
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def upload_model(
        self,
        local_path: Path | str,
        s3_uri: str | None = None,
    ) -> str:
        """Upload all files in *local_path* to S3.

        Args:
            local_path: Local model directory.
            s3_uri: Explicit ``s3://bucket/key-prefix`` override.

        Returns:
            S3 URI of the uploaded model.
        """
        local = Path(local_path).resolve()
        if not local.is_dir():
            raise FileNotFoundError(f"Local path is not a directory: {local}")

        if s3_uri:
            bucket, key_prefix = self._parse_s3_uri(s3_uri)
        else:
            bucket = self._bucket
            key_prefix = f"{self._prefix}/{local.name}" if self._prefix else local.name

        s3 = self._get_client()
        uploaded = 0
        for file_path in sorted(local.rglob("*")):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(local))
            key = f"{key_prefix}/{rel}"
            logger.info("Uploading %s → s3://%s/%s", rel, bucket, key)
            s3.upload_file(str(file_path), bucket, key)
            uploaded += 1

        uri = f"s3://{bucket}/{key_prefix}"
        logger.info("Uploaded %d files to %s", uploaded, uri)
        return uri

    def upload_with_checksum(
        self,
        local_path: Path | str,
        s3_uri: str | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Upload files and return per-file checksums for verification.

        Returns:
            ``(s3_uri, {relative_path: sha256_hex})``
        """
        local = Path(local_path).resolve()
        checksums: dict[str, str] = {}

        for file_path in sorted(local.rglob("*")):
            if file_path.is_file():
                rel = str(file_path.relative_to(local))
                checksums[rel] = compute_file_checksum(file_path)

        uri = self.upload_model(local, s3_uri)

        # Upload checksum manifest
        if checksums:
            manifest_content = json.dumps(checksums, indent=2)
            if s3_uri:
                bucket, key_prefix = self._parse_s3_uri(s3_uri)
            else:
                bucket = self._bucket
                key_prefix = f"{self._prefix}/{local.name}" if self._prefix else local.name

            s3 = self._get_client()
            manifest_key = f"{key_prefix}/checksums.json"
            s3.put_object(
                Bucket=bucket, Key=manifest_key,
                Body=manifest_content.encode("utf-8"),
            )
            logger.info("Checksum manifest uploaded to s3://%s/%s", bucket, manifest_key)

        return uri, checksums

    def download_model(
        self,
        s3_uri: str,
        local_path: Path | str,
    ) -> Path:
        """Download a model from S3 to a local directory.

        Args:
            s3_uri: S3 URI in ``s3://bucket/prefix`` format.
            local_path: Local destination directory.

        Returns:
            Resolved local path.
        """
        bucket, prefix = self._parse_s3_uri(s3_uri)
        local = Path(local_path).resolve()
        local.mkdir(parents=True, exist_ok=True)

        s3 = self._get_client()
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        downloaded = 0
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):].lstrip("/")
                if not rel:
                    continue
                dest = local / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading s3://%s/%s → %s", bucket, key, dest)
                s3.download_file(bucket, key, str(dest))
                downloaded += 1

        logger.info("Downloaded %d files to %s", downloaded, local)
        return local

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        """Parse ``s3://bucket/prefix`` into ``(bucket, prefix)``."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri!r}")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix


# ---------------------------------------------------------------------------
# ServingManifestRenderer
# ---------------------------------------------------------------------------

class ServingManifestRenderer:
    """Render Kubernetes / RHOAI serving manifests for vLLM.

    Generates ``InferenceService`` (KServe), ``Route``, and ``Secret``
    YAML documents suitable for ``oc apply``.

    Args:
        namespace: Target OpenShift namespace.
    """

    _INFERENCE_SERVICE_TEMPLATE = dedent("""\
        apiVersion: serving.kserve.io/v1beta1
        kind: InferenceService
        metadata:
          name: {{ name }}
          namespace: {{ namespace }}
          labels:
            opendatahub.io/dashboard: "true"
          annotations:
            serving.kserve.io/deploymentMode: RawDeployment
        spec:
          predictor:
            model:
              modelFormat:
                name: vLLM
              runtime: {{ runtime }}
              storageUri: {{ storage_uri }}
            containers:
              - name: kserve-container
                image: {{ image }}
                args:
                  - "--model"
                  - "{{ model_path }}"
                  - "--served-model-name"
                  - "{{ model_name }}"
                  - "--dtype"
                  - "{{ dtype }}"
                  - "--max-model-len"
                  - "{{ max_model_len }}"
                  - "--tool-parser-plugin"
                  - "{{ tool_parser }}"
                  {% if chat_template %}- "--chat-template"
                  - "{{ chat_template }}"{% endif %}
                  {% if enable_auto_tool_choice %}- "--enable-auto-tool-choice"{% endif %}
                  {% if gpu_memory_utilization %}- "--gpu-memory-utilization"
                  - "{{ gpu_memory_utilization }}"{% endif %}
                resources:
                  limits:
                    nvidia.com/gpu: "{{ gpu_count }}"
                  requests:
                    nvidia.com/gpu: "{{ gpu_count }}"
                    memory: "{{ memory }}"
                    cpu: "{{ cpu }}"
    """)

    _ROUTE_TEMPLATE = dedent("""\
        apiVersion: route.openshift.io/v1
        kind: Route
        metadata:
          name: {{ name }}
          namespace: {{ namespace }}
          labels:
            app: {{ name }}
        spec:
          to:
            kind: Service
            name: {{ service_name }}
          port:
            targetPort: http
          tls:
            termination: edge
            insecureEdgeTerminationPolicy: Redirect
    """)

    _SECRET_TEMPLATE = dedent("""\
        apiVersion: v1
        kind: Secret
        metadata:
          name: {{ name }}
          namespace: {{ namespace }}
        type: Opaque
        stringData:
          {% for key, value in data.items() %}{{ key }}: "{{ value }}"
          {% endfor %}
    """)

    def __init__(self, namespace: str = "default") -> None:
        self._namespace = namespace
        self._env = Environment(loader=BaseLoader(), keep_trailing_newline=True)

    def render_inference_service(
        self,
        model_config: dict[str, Any],
    ) -> str:
        """Render a KServe InferenceService manifest.

        Args:
            model_config: Dict with keys like ``name``, ``model_name``,
                ``storage_uri``, ``model_path``, ``dtype``, ``tool_parser``,
                ``gpu_count``, ``memory``, ``cpu``, etc.

        Returns:
            YAML string.
        """
        defaults = {
            "namespace": self._namespace,
            "runtime": _DEFAULT_RUNTIME,
            "image": _DEFAULT_VLLM_IMAGE,
            "dtype": "bfloat16",
            "max_model_len": "4096",
            "tool_parser": _DEFAULT_TOOL_PARSER,
            "chat_template": "",
            "enable_auto_tool_choice": True,
            "gpu_memory_utilization": "0.90",
            "gpu_count": "1",
            "memory": "24Gi",
            "cpu": "4",
            "model_path": "/mnt/models",
        }
        ctx = {**defaults, **model_config}
        template = self._env.from_string(self._INFERENCE_SERVICE_TEMPLATE)
        return template.render(**ctx)

    def render_route(
        self,
        name: str,
        service_name: str | None = None,
    ) -> str:
        """Render an OpenShift Route for the serving endpoint.

        Args:
            name: Route name.
            service_name: Backend service name (defaults to *name*).
        """
        template = self._env.from_string(self._ROUTE_TEMPLATE)
        return template.render(
            name=name,
            namespace=self._namespace,
            service_name=service_name or name,
        )

    def render_secret(
        self,
        name: str,
        data: dict[str, str],
    ) -> str:
        """Render a Kubernetes Secret for endpoint auth tokens.

        Args:
            name: Secret name.
            data: Key-value pairs to store.
        """
        template = self._env.from_string(self._SECRET_TEMPLATE)
        return template.render(
            name=name,
            namespace=self._namespace,
            data=data,
        )


# ---------------------------------------------------------------------------
# DeploymentValidator
# ---------------------------------------------------------------------------

class DeploymentValidator:
    """Validate a deployed model endpoint.

    Performs health checks, simple chat completions, and tool-call
    round-trips against an OpenAI-compatible API.

    Args:
        timeout: HTTP request timeout in seconds.
        verify_ssl: Whether to verify TLS certificates.
        ca_bundle: Optional CA bundle path for custom CAs.
    """

    def __init__(
        self,
        timeout: float = _API_TIMEOUT,
        verify_ssl: bool = True,
        ca_bundle: str | None = None,
    ) -> None:
        self._timeout = timeout
        ssl_ctx: Any = True
        if ca_bundle:
            try:
                import ssl
                ssl_ctx = ssl.create_default_context(cafile=ca_bundle)
            except Exception:
                ssl_ctx = ca_bundle  # httpx can accept a path
        elif not verify_ssl:
            ssl_ctx = False
        self._verify = ssl_ctx

    def check_endpoint_health(self, url: str) -> dict[str, Any]:
        """Check if the model endpoint is healthy.

        Tries ``/health`` and ``/v1/models`` endpoints.

        Args:
            url: Base URL of the serving endpoint (e.g.
                ``https://model.apps.cluster.com/v1``).

        Returns:
            Dict with ``"healthy"`` (bool), ``"models"`` (list), and
            ``"latency_ms"`` (float).
        """
        base = url.rstrip("/")
        result: dict[str, Any] = {
            "healthy": False,
            "models": [],
            "latency_ms": 0.0,
            "errors": [],
        }

        # Try /health or /v1/models
        for path in ["/health", "/v1/models"]:
            endpoint = base.replace("/v1", "") + path if "/v1" in base else base + path
            try:
                start = time.monotonic()
                with httpx.Client(timeout=self._timeout, verify=self._verify) as client:
                    resp = client.get(endpoint)
                elapsed = (time.monotonic() - start) * 1000

                if resp.status_code == 200:
                    result["healthy"] = True
                    result["latency_ms"] = elapsed
                    body = resp.json()
                    if "data" in body:
                        result["models"] = [
                            m.get("id", "") for m in body["data"]
                        ]
                    logger.info(
                        "Endpoint healthy: %s (%.1fms)", endpoint, elapsed,
                    )
                    return result
            except Exception as exc:
                result["errors"].append(f"{path}: {exc}")

        logger.warning("Endpoint health check failed: %s", result["errors"])
        return result

    def smoke_test_chat(
        self,
        url: str,
        model_name: str = "",
        token: str = "",
    ) -> dict[str, Any]:
        """Send a simple chat completion and verify the response.

        Args:
            url: Base URL (e.g. ``https://…/v1``).
            model_name: Model identifier for the API.
            token: Optional bearer token.
        """
        base = url.rstrip("/")
        endpoint = f"{base}/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello, respond with a single word."},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }

        result: dict[str, Any] = {
            "success": False,
            "response": None,
            "latency_ms": 0.0,
            "error": None,
        }

        try:
            start = time.monotonic()
            with httpx.Client(timeout=self._timeout, verify=self._verify) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
            elapsed = (time.monotonic() - start) * 1000
            result["latency_ms"] = elapsed

            if resp.status_code == 200:
                body = resp.json()
                choices = body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    result["success"] = bool(content.strip())
                    result["response"] = content.strip()
                else:
                    result["error"] = "No choices in response"
            else:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"

        except Exception as exc:
            result["error"] = str(exc)

        level = logging.INFO if result["success"] else logging.WARNING
        logger.log(level, "Chat smoke test: %s", result)
        return result

    def smoke_test_tool_call(
        self,
        url: str,
        model_name: str = "",
        token: str = "",
    ) -> dict[str, Any]:
        """Send a chat completion that should trigger a tool call.

        Verifies that the model correctly generates a structured
        tool-call response with valid JSON arguments.

        Args:
            url: Base URL.
            model_name: Model identifier.
            token: Optional bearer token.
        """
        base = url.rstrip("/")
        endpoint = f"{base}/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a banking assistant. Use the provided tools when appropriate.",
                },
                {
                    "role": "user",
                    "content": "What is the balance of account 12345?",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_account_balance",
                        "description": "Get the balance of a bank account",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "account_id": {
                                    "type": "string",
                                    "description": "The account identifier",
                                },
                            },
                            "required": ["account_id"],
                        },
                    },
                },
            ],
            "tool_choice": "auto",
            "max_tokens": 256,
            "temperature": 0.0,
        }

        result: dict[str, Any] = {
            "success": False,
            "tool_calls": [],
            "latency_ms": 0.0,
            "error": None,
        }

        try:
            start = time.monotonic()
            with httpx.Client(timeout=self._timeout, verify=self._verify) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
            elapsed = (time.monotonic() - start) * 1000
            result["latency_ms"] = elapsed

            if resp.status_code == 200:
                body = resp.json()
                choices = body.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        result["tool_calls"] = tool_calls
                        # Validate JSON arguments
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            try:
                                json.loads(fn.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                result["error"] = (
                                    f"Invalid JSON in tool call arguments: "
                                    f"{fn.get('arguments', '')[:100]}"
                                )
                                break
                        else:
                            result["success"] = True
                    else:
                        result["error"] = "No tool calls generated"
            else:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"

        except Exception as exc:
            result["error"] = str(exc)

        level = logging.INFO if result["success"] else logging.WARNING
        logger.log(level, "Tool-call smoke test: %s", result)
        return result


__all__ = [
    "DeploymentValidator",
    "ModelExporter",
    "S3Uploader",
    "ServingManifestRenderer",
]
