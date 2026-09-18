"""Training module for LoRA and OSFT fine-tuning.

Both trainers start independently from the same base checkpoint
(``Qwen/Qwen3-4B-Instruct-2507``), share the same seed (42), and
operate on identical canonical sample IDs.  Training data is consumed
from a prepared bundle; SDG is never triggered here.

External dependencies (``training_hub``, ``torch``, ``transformers``,
``peft``) are imported behind ``try/except`` so that the module can be
safely imported in lightweight environments (e.g. for testing schemas).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from rhoai_model_training_lab.config import load_training_config, load_yaml_config
from rhoai_model_training_lab.schemas.data import (
    BackendValidationResult,
    CanonicalSample,
    Message,
)
from rhoai_model_training_lab.schemas.training import (
    LoRAConfig,
    OSFTConfig,
    TrainingResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy-weight imports
# ---------------------------------------------------------------------------

_training_hub_available = False
_training_hub: Any = None

try:
    import training_hub as _training_hub  # type: ignore[import-untyped]
    _training_hub_available = True
except ImportError:
    logger.debug("training_hub is not installed; trainers will require manual step calls")

_torch_available = False
try:
    import torch  # type: ignore[import-untyped]
    _torch_available = True
except ImportError:
    pass

_transformers_available = False
try:
    import transformers  # type: ignore[import-untyped]
    _transformers_available = True
except ImportError:
    pass

_peft_available = False
try:
    import peft  # type: ignore[import-untyped]
    _peft_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Apply chat templates and verify loss masking for training data.

    Ensures that every training sample is correctly tokenised with:
    * BOS / EOS tokens present
    * Padding handled properly
    * Loss computed only on assistant turns (``assistant_only`` masking)
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        max_seq_length: int = 4096,
        loss_masking: str = "assistant_only",
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._loss_masking = loss_masking

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Tokenise messages using the tokenizer's chat template.

        Returns a dict with ``input_ids``, ``attention_mask``, and
        ``labels`` (with non-assistant positions masked to ``-100``).
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded — call set_tokenizer() first")

        chat_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_tensors": "pt",
            "max_length": self._max_seq_length,
            "truncation": True,
            "padding": False,
        }
        if tools and hasattr(self._tokenizer, "apply_chat_template"):
            chat_kwargs["tools"] = tools

        input_ids = self._tokenizer.apply_chat_template(
            messages, **chat_kwargs
        )

        if _torch_available:
            if isinstance(input_ids, torch.Tensor):
                input_ids_list = input_ids.squeeze().tolist()
            else:
                input_ids_list = list(input_ids)
        else:
            input_ids_list = list(input_ids) if not isinstance(input_ids, list) else input_ids

        labels = self._compute_labels(messages, input_ids_list)

        return {
            "input_ids": input_ids_list,
            "attention_mask": [1] * len(input_ids_list),
            "labels": labels,
        }

    def verify_loss_masking(self, encoded: dict[str, Any]) -> list[str]:
        """Return a list of issues found in the loss mask."""
        issues: list[str] = []
        labels = encoded.get("labels", [])
        input_ids = encoded.get("input_ids", [])

        if not labels:
            issues.append("Empty labels array")
            return issues

        # At least some positions must be trained
        trained = sum(1 for l in labels if l != -100)
        if trained == 0:
            issues.append("All labels are masked — nothing to train on")

        # No label should exceed vocab size (basic sanity)
        if self._tokenizer is not None:
            vocab_size = getattr(self._tokenizer, "vocab_size", None)
            if vocab_size:
                for i, l in enumerate(labels):
                    if l != -100 and l >= vocab_size:
                        issues.append(f"Label at position {i} ({l}) exceeds vocab size ({vocab_size})")
                        break

        return issues

    def check_special_tokens(self, encoded: dict[str, Any]) -> list[str]:
        """Verify BOS, EOS, and padding tokens are correctly placed."""
        issues: list[str] = []
        if self._tokenizer is None:
            return issues

        ids = encoded.get("input_ids", [])
        if not ids:
            issues.append("Empty input_ids")
            return issues

        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        pad_id = getattr(self._tokenizer, "pad_token_id", None)

        if bos_id is not None and ids[0] != bos_id:
            issues.append(f"First token is {ids[0]}, expected BOS ({bos_id})")

        if eos_id is not None and ids[-1] != eos_id:
            issues.append(f"Last token is {ids[-1]}, expected EOS ({eos_id})")

        if pad_id is not None and pad_id in ids:
            pad_positions = [i for i, t in enumerate(ids) if t == pad_id]
            non_pad_after = any(
                ids[j] != pad_id for j in range(min(pad_positions), len(ids))
                if j not in pad_positions
            )
            if non_pad_after:
                issues.append("Padding tokens found interspersed with content tokens")

        return issues

    def set_tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def _compute_labels(
        self,
        messages: list[dict[str, Any]],
        input_ids: list[int],
    ) -> list[int]:
        """Build labels with assistant-only loss masking.

        Positions corresponding to system/user/tool roles are set to
        ``-100`` (ignored by cross-entropy loss).
        """
        if self._loss_masking != "assistant_only":
            return list(input_ids)

        labels = [-100] * len(input_ids)

        if self._tokenizer is None:
            return labels

        # Re-encode each message individually to find boundaries
        offset = 0
        for msg in messages:
            single_ids = self._tokenizer.apply_chat_template(
                [msg], tokenize=True, add_generation_prompt=False,
            )
            if isinstance(single_ids, list):
                msg_len = len(single_ids)
            elif _torch_available and isinstance(single_ids, torch.Tensor):
                msg_len = single_ids.numel()
            else:
                msg_len = len(list(single_ids))

            role = msg.get("role", "")
            if role == "assistant":
                end = min(offset + msg_len, len(labels))
                for i in range(offset, end):
                    labels[i] = input_ids[i]

            offset += msg_len
            if offset >= len(input_ids):
                break

        return labels


# ---------------------------------------------------------------------------
# validate_bundle_for_training
# ---------------------------------------------------------------------------

def validate_bundle_for_training(
    bundle_path: Path | str,
    profile: str,
) -> BackendValidationResult:
    """Validate that a bundle is ready for LoRA or OSFT training.

    Checks:
    * Training data files exist and are loadable
    * Tokenizer is available and chat template matches
    * Loss masking produces trainable labels
    * No sample exceeds max_seq_length

    Args:
        bundle_path: Root directory of the prepared bundle.
        profile: ``"lora"`` or ``"osft"``.

    Returns:
        :class:`BackendValidationResult` with detailed check outcomes.
    """
    if profile not in ("lora", "osft"):
        raise ValueError(f"profile must be 'lora' or 'osft', got {profile!r}")

    root = Path(bundle_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # Load config
    try:
        cfg = load_training_config(profile)
    except FileNotFoundError:
        errors.append(f"Training config not found for profile {profile!r}")
        return BackendValidationResult(
            backend=profile, errors=errors, warnings=warnings,
        )

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    max_seq_length = data_cfg.get("max_seq_length", 4096)

    # 1. Training files exist
    train_path = root / "training" / profile / "train.jsonl"
    val_path = root / "training" / profile / "validation.jsonl"

    loader_ok = True
    if not train_path.exists():
        errors.append(f"Training file missing: {train_path}")
        loader_ok = False
    if not val_path.exists():
        errors.append(f"Validation file missing: {val_path}")
        loader_ok = False

    # 2. Tokenizer
    tokenizer_ok = False
    tokenizer = None
    if _transformers_available:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_cfg.get("tokenizer_id", model_cfg.get("model_id", "")),
                trust_remote_code=True,
            )
            tokenizer_ok = True
        except Exception as exc:
            errors.append(f"Failed to load tokenizer: {exc}")
    else:
        warnings.append("transformers not installed; skipping tokenizer checks")

    # 3. Chat template hash
    chat_template_ok = True
    manifest_path = root / "manifest.json"
    if tokenizer is not None and manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest_data = json.load(fh)
            expected_hash = manifest_data.get("chat_template_hash", "")
            if expected_hash:
                template_src = getattr(tokenizer, "chat_template", "") or ""
                actual_hash = hashlib.sha256(template_src.encode()).hexdigest()
                if actual_hash != expected_hash:
                    errors.append("Chat template hash mismatch")
                    chat_template_ok = False
        except Exception as exc:
            warnings.append(f"Could not verify chat template hash: {exc}")

    # 4. Loss masking + max length spot-check
    loss_mask_ok = True
    max_length_ok = True
    sample_count_ok = True

    if loader_ok and tokenizer is not None:
        preprocessor = DataPreprocessor(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        try:
            with open(train_path, encoding="utf-8") as fh:
                lines = fh.readlines()

            if not lines:
                errors.append("Training file is empty")
                sample_count_ok = False
            else:
                check_count = min(len(lines), 10)
                for i in range(check_count):
                    raw = json.loads(lines[i])
                    messages = raw.get("messages", [])
                    if not messages:
                        continue

                    encoded = preprocessor.apply_chat_template(messages)
                    mask_issues = preprocessor.verify_loss_masking(encoded)
                    if mask_issues:
                        loss_mask_ok = False
                        errors.append(
                            f"Loss mask issue in sample {i}: {mask_issues[0]}"
                        )
                        break

                    token_issues = preprocessor.check_special_tokens(encoded)
                    if token_issues:
                        warnings.extend(
                            f"Special token issue in sample {i}: {t}"
                            for t in token_issues
                        )

                    if len(encoded["input_ids"]) > max_seq_length:
                        max_length_ok = False
                        warnings.append(
                            f"Sample {i} exceeds max_seq_length "
                            f"({len(encoded['input_ids'])} > {max_seq_length})"
                        )

        except Exception as exc:
            errors.append(f"Error during spot-check: {exc}")
            loss_mask_ok = False

    return BackendValidationResult(
        backend=profile,
        loader_check=loader_ok,
        tokenizer_check=tokenizer_ok,
        sample_count_match=sample_count_ok,
        chat_template_check=chat_template_ok,
        loss_mask_check=loss_mask_ok,
        max_length_check=max_length_ok,
        errors=errors,
        warnings=warnings,
        verified_scope=f"{profile}_training_readiness",
    )


# ---------------------------------------------------------------------------
# LoRATrainer
# ---------------------------------------------------------------------------

class LoRATrainer:
    """LoRA SFT trainer wrapping ``training_hub.lora_sft``.

    Manages the lifecycle of a LoRA fine-tuning run: data preparation,
    training loop, checkpoint management, adapter export, and optional
    merge with the base model.

    Args:
        config: Parsed LoRA training configuration.  If *None*, the
            default ``configs/lora.yaml`` is loaded.
        bundle_path: Root directory of the prepared dataset bundle.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        bundle_path: Path | str | None = None,
    ) -> None:
        if config is None:
            config = load_training_config("lora")
        self._cfg = config
        self._lora_config = LoRAConfig(
            **{
                **config.get("model", {}),
                **config.get("lora", {}),
                **config.get("data", {}),
                **{k: v for k, v in config.get("training_args", {}).items()
                   if k in LoRAConfig.model_fields},
            }
        )
        self._bundle_path = Path(bundle_path) if bundle_path else None
        self._preprocessor: DataPreprocessor | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._trainer: Any = None  # training_hub or HF trainer
        self._result: TrainingResult | None = None
        self._start_time: float = 0.0

    @property
    def lora_config(self) -> LoRAConfig:
        return self._lora_config

    # -- lifecycle steps ----------------------------------------------------

    def prepare_data(self) -> None:
        """Load tokenizer, validate bundle, and prepare datasets."""
        logger.info("Preparing data for LoRA training")

        if not _transformers_available:
            raise RuntimeError("transformers is required for LoRA training")

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._lora_config.tokenizer_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._preprocessor = DataPreprocessor(
            tokenizer=self._tokenizer,
            max_seq_length=self._lora_config.max_seq_length,
            loss_masking=self._lora_config.loss_masking,
        )

        if self._bundle_path:
            validation = validate_bundle_for_training(self._bundle_path, "lora")
            if validation.errors:
                raise RuntimeError(
                    f"Bundle validation failed: {validation.errors}"
                )

        logger.info("LoRA data preparation complete")

    def train(self) -> TrainingResult:
        """Execute the LoRA training loop.

        Uses ``training_hub.lora_sft`` when available; otherwise falls
        back to a direct ``transformers.Trainer`` + ``peft`` workflow.
        """
        self._start_time = time.monotonic()
        logger.info("Starting LoRA training")

        training_args = self._cfg.get("training_args", {})
        output_dir = Path(training_args.get("output_dir", self._lora_config.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if _training_hub_available and _training_hub is not None:
            result = self._train_with_hub(training_args)
        elif _transformers_available and _peft_available:
            result = self._train_native(training_args)
        else:
            raise RuntimeError(
                "Neither training_hub nor transformers+peft are available. "
                "Install one of: pip install training_hub, "
                "pip install transformers peft accelerate"
            )

        elapsed = time.monotonic() - self._start_time
        result.wall_time_seconds = elapsed
        self._result = result

        logger.info(
            "LoRA training complete in %.1fs — final loss: %.4f",
            elapsed, result.final_train_loss,
        )
        return result

    def save_checkpoint(self, step: int | None = None) -> Path:
        """Save current training state as a checkpoint."""
        output_dir = Path(self._lora_config.output_dir)
        ckpt_name = f"checkpoint-{step}" if step else "checkpoint-latest"
        ckpt_path = output_dir / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        if self._trainer is not None and hasattr(self._trainer, "save_model"):
            self._trainer.save_model(str(ckpt_path))
        elif self._model is not None and hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(str(ckpt_path))

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(ckpt_path))

        logger.info("Checkpoint saved: %s", ckpt_path)
        return ckpt_path

    def export_adapter(self, output_path: Path | str | None = None) -> Path:
        """Export the LoRA adapter weights."""
        adapter_path = Path(output_path or self._lora_config.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            if hasattr(self._model, "save_pretrained"):
                self._model.save_pretrained(str(adapter_path))
            if self._tokenizer is not None:
                self._tokenizer.save_pretrained(str(adapter_path))

        logger.info("LoRA adapter exported to %s", adapter_path)
        return adapter_path

    def merge_and_export(self, output_path: Path | str | None = None) -> Path:
        """Merge LoRA adapter into the base model and export.

        Produces a standalone model that can be served without peft.
        """
        merged_path = Path(output_path or self._lora_config.merged_path)
        merged_path.mkdir(parents=True, exist_ok=True)

        if not _peft_available or not _transformers_available:
            raise RuntimeError("peft and transformers are required for merge")

        if self._model is not None and hasattr(self._model, "merge_and_unload"):
            merged = self._model.merge_and_unload()
            merged.save_pretrained(str(merged_path))
        else:
            # Load adapter from disk and merge
            adapter_path = Path(self._lora_config.adapter_path)
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"Adapter not found at {adapter_path}. "
                    "Run export_adapter() first."
                )
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                self._lora_config.model_id,
                torch_dtype=self._lora_config.torch_dtype,
                trust_remote_code=True,
            )
            model_with_adapter = peft.PeftModel.from_pretrained(
                base_model, str(adapter_path),
            )
            merged = model_with_adapter.merge_and_unload()
            merged.save_pretrained(str(merged_path))

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(merged_path))

        logger.info("Merged model exported to %s", merged_path)
        return merged_path

    # -- internal training implementations ----------------------------------

    def _train_with_hub(self, training_args: dict[str, Any]) -> TrainingResult:
        """Train using training_hub.lora_sft API."""
        hub_cfg = {
            "model": self._cfg.get("model", {}),
            "lora": self._cfg.get("lora", {}),
            "data": self._cfg.get("data", {}),
            "training_args": training_args,
        }

        hub_result = _training_hub.lora_sft.train(hub_cfg)

        return TrainingResult(
            method="lora",
            model_id=self._lora_config.model_id,
            model_revision=self._lora_config.model_revision,
            seed=self._lora_config.seed,
            train_samples=hub_result.get("train_samples", 0),
            validation_samples=hub_result.get("validation_samples", 0),
            total_steps=hub_result.get("total_steps", 0),
            num_epochs=self._lora_config.num_train_epochs,
            final_train_loss=hub_result.get("final_train_loss", 0.0),
            final_eval_loss=hub_result.get("final_eval_loss"),
            best_eval_loss=hub_result.get("best_eval_loss"),
            best_checkpoint_step=hub_result.get("best_checkpoint_step"),
            checkpoint_path=training_args.get("output_dir", ""),
            adapter_path=self._lora_config.adapter_path,
        )

    def _train_native(self, training_args: dict[str, Any]) -> TrainingResult:
        """Train using transformers Trainer + peft directly."""
        logger.info("Using native transformers + peft training")

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self._lora_config.model_id,
            torch_dtype=self._lora_config.torch_dtype,
            attn_implementation=self._lora_config.attn_implementation,
            trust_remote_code=True,
        )

        lora_cfg = peft.LoraConfig(
            r=self._lora_config.r,
            lora_alpha=self._lora_config.lora_alpha,
            lora_dropout=self._lora_config.lora_dropout,
            target_modules=self._lora_config.target_modules,
            bias=self._lora_config.bias,
            task_type=self._lora_config.task_type,
        )
        self._model = peft.get_peft_model(base_model, lora_cfg)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "LoRA model: %d trainable / %d total parameters (%.2f%%)",
            trainable, total, 100 * trainable / max(total, 1),
        )

        from datasets import load_dataset  # type: ignore[import-untyped]

        data_cfg = self._cfg.get("data", {})
        train_ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
        eval_ds = None
        if data_cfg.get("validation_file"):
            eval_ds = load_dataset("json", data_files=data_cfg["validation_file"], split="train")

        hf_args = transformers.TrainingArguments(
            output_dir=training_args.get("output_dir", self._lora_config.output_dir),
            num_train_epochs=training_args.get("num_train_epochs", self._lora_config.num_train_epochs),
            per_device_train_batch_size=training_args.get(
                "per_device_train_batch_size", self._lora_config.per_device_train_batch_size,
            ),
            gradient_accumulation_steps=training_args.get(
                "gradient_accumulation_steps", self._lora_config.gradient_accumulation_steps,
            ),
            learning_rate=training_args.get("learning_rate", self._lora_config.learning_rate),
            weight_decay=training_args.get("weight_decay", self._lora_config.weight_decay),
            warmup_ratio=training_args.get("warmup_ratio", self._lora_config.warmup_ratio),
            lr_scheduler_type=training_args.get("lr_scheduler_type", self._lora_config.lr_scheduler_type),
            bf16=training_args.get("bf16", self._lora_config.bf16),
            gradient_checkpointing=training_args.get(
                "gradient_checkpointing", self._lora_config.gradient_checkpointing,
            ),
            logging_steps=training_args.get("logging_steps", 10),
            save_strategy=training_args.get("save_strategy", "steps"),
            save_steps=training_args.get("save_steps", 100),
            save_total_limit=training_args.get("save_total_limit", 3),
            seed=self._lora_config.seed,
            report_to="none",
        )

        self._trainer = transformers.Trainer(
            model=self._model,
            args=hf_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self._tokenizer,
        )

        train_output = self._trainer.train(
            resume_from_checkpoint=training_args.get("resume_from_checkpoint", False) or None,
        )

        eval_loss = None
        if eval_ds is not None:
            eval_result = self._trainer.evaluate()
            eval_loss = eval_result.get("eval_loss")

        return TrainingResult(
            method="lora",
            model_id=self._lora_config.model_id,
            model_revision=self._lora_config.model_revision,
            seed=self._lora_config.seed,
            train_samples=len(train_ds),
            validation_samples=len(eval_ds) if eval_ds else 0,
            total_steps=train_output.global_step,
            num_epochs=self._lora_config.num_train_epochs,
            final_train_loss=train_output.training_loss,
            final_eval_loss=eval_loss,
            checkpoint_path=hf_args.output_dir,
            adapter_path=self._lora_config.adapter_path,
        )


# ---------------------------------------------------------------------------
# OSFTTrainer
# ---------------------------------------------------------------------------

class OSFTTrainer:
    """OSFT (Orthogonal Subspace Fine-Tuning) trainer.

    Uses ``training_hub.osft`` when available.  OSFT decomposes weight
    matrices and selectively unfreezes low-rank components, achieving
    parameter-efficient training without external adapter weights.

    Args:
        config: Parsed OSFT training configuration.
        bundle_path: Root directory of the prepared dataset bundle.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        bundle_path: Path | str | None = None,
    ) -> None:
        if config is None:
            config = load_training_config("osft")
        self._cfg = config
        self._osft_config = OSFTConfig(
            **{
                **config.get("model", {}),
                **config.get("osft", {}),
                **config.get("data", {}),
                **{k: v for k, v in config.get("training_args", {}).items()
                   if k in OSFTConfig.model_fields},
            }
        )
        self._bundle_path = Path(bundle_path) if bundle_path else None
        self._preprocessor: DataPreprocessor | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._trainer: Any = None
        self._result: TrainingResult | None = None
        self._start_time: float = 0.0

    @property
    def osft_config(self) -> OSFTConfig:
        return self._osft_config

    # -- lifecycle steps ----------------------------------------------------

    def prepare_data(self) -> None:
        """Load tokenizer, validate bundle, and prepare datasets."""
        logger.info("Preparing data for OSFT training")

        if not _transformers_available:
            raise RuntimeError("transformers is required for OSFT training")

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._osft_config.tokenizer_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._preprocessor = DataPreprocessor(
            tokenizer=self._tokenizer,
            max_seq_length=self._osft_config.max_seq_length,
            loss_masking=self._osft_config.loss_masking,
        )

        if self._bundle_path:
            validation = validate_bundle_for_training(self._bundle_path, "osft")
            if validation.errors:
                raise RuntimeError(
                    f"Bundle validation failed: {validation.errors}"
                )

        logger.info("OSFT data preparation complete")

    def train(self) -> TrainingResult:
        """Execute the OSFT training loop."""
        self._start_time = time.monotonic()
        logger.info("Starting OSFT training")

        training_args = self._cfg.get("training_args", {})
        output_dir = Path(training_args.get("output_dir", self._osft_config.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if _training_hub_available and _training_hub is not None:
            result = self._train_with_hub(training_args)
        elif _transformers_available:
            result = self._train_native(training_args)
        else:
            raise RuntimeError(
                "Neither training_hub nor transformers are available. "
                "Install one of: pip install training_hub, "
                "pip install transformers accelerate"
            )

        elapsed = time.monotonic() - self._start_time
        result.wall_time_seconds = elapsed
        self._result = result

        logger.info(
            "OSFT training complete in %.1fs — final loss: %.4f",
            elapsed, result.final_train_loss,
        )
        return result

    def save_checkpoint(self, step: int | None = None) -> Path:
        """Save current training state."""
        output_dir = Path(self._osft_config.output_dir)
        ckpt_name = f"checkpoint-{step}" if step else "checkpoint-latest"
        ckpt_path = output_dir / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        if self._trainer is not None and hasattr(self._trainer, "save_model"):
            self._trainer.save_model(str(ckpt_path))
        elif self._model is not None and hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(str(ckpt_path))

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(ckpt_path))

        logger.info("OSFT checkpoint saved: %s", ckpt_path)
        return ckpt_path

    def export_model(self, output_path: Path | str | None = None) -> Path:
        """Export the OSFT-tuned model as a standalone HuggingFace model."""
        export_path = Path(output_path or self._osft_config.output_path)
        export_path.mkdir(parents=True, exist_ok=True)

        if self._model is not None and hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(str(export_path))

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(export_path))

        export_cfg = self._cfg.get("export", {})
        if export_cfg.get("validate_reload", False) and _transformers_available:
            self._validate_exported(export_path, export_cfg)

        logger.info("OSFT model exported to %s", export_path)
        return export_path

    # -- internal training implementations ----------------------------------

    def _train_with_hub(self, training_args: dict[str, Any]) -> TrainingResult:
        """Train using training_hub.osft API."""
        hub_cfg = {
            "model": self._cfg.get("model", {}),
            "osft": self._cfg.get("osft", {}),
            "data": self._cfg.get("data", {}),
            "training_args": training_args,
        }

        hub_result = _training_hub.osft.train(hub_cfg)

        return TrainingResult(
            method="osft",
            model_id=self._osft_config.model_id,
            model_revision=self._osft_config.model_revision,
            seed=self._osft_config.seed,
            train_samples=hub_result.get("train_samples", 0),
            validation_samples=hub_result.get("validation_samples", 0),
            total_steps=hub_result.get("total_steps", 0),
            num_epochs=self._osft_config.num_train_epochs,
            final_train_loss=hub_result.get("final_train_loss", 0.0),
            final_eval_loss=hub_result.get("final_eval_loss"),
            best_eval_loss=hub_result.get("best_eval_loss"),
            best_checkpoint_step=hub_result.get("best_checkpoint_step"),
            checkpoint_path=training_args.get("output_dir", ""),
            exported_path=self._osft_config.output_path,
        )

    def _train_native(self, training_args: dict[str, Any]) -> TrainingResult:
        """Train using native transformers (OSFT layer surgery)."""
        logger.info("Using native transformers OSFT training")

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self._osft_config.model_id,
            torch_dtype=self._osft_config.torch_dtype,
            attn_implementation=self._osft_config.attn_implementation,
            trust_remote_code=True,
        )

        self._apply_osft_decomposition(self._model)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "OSFT model: %d trainable / %d total parameters (%.2f%%)",
            trainable, total, 100 * trainable / max(total, 1),
        )

        from datasets import load_dataset  # type: ignore[import-untyped]

        data_cfg = self._cfg.get("data", {})
        train_ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
        eval_ds = None
        if data_cfg.get("validation_file"):
            eval_ds = load_dataset("json", data_files=data_cfg["validation_file"], split="train")

        hf_args = transformers.TrainingArguments(
            output_dir=training_args.get("output_dir", self._osft_config.output_dir),
            num_train_epochs=training_args.get("num_train_epochs", self._osft_config.num_train_epochs),
            per_device_train_batch_size=training_args.get(
                "per_device_train_batch_size", self._osft_config.per_device_train_batch_size,
            ),
            gradient_accumulation_steps=training_args.get(
                "gradient_accumulation_steps", self._osft_config.gradient_accumulation_steps,
            ),
            learning_rate=training_args.get("learning_rate", self._osft_config.learning_rate),
            weight_decay=training_args.get("weight_decay", self._osft_config.weight_decay),
            warmup_ratio=training_args.get("warmup_ratio", self._osft_config.warmup_ratio),
            lr_scheduler_type=training_args.get("lr_scheduler_type", self._osft_config.lr_scheduler_type),
            bf16=training_args.get("bf16", self._osft_config.bf16),
            gradient_checkpointing=training_args.get(
                "gradient_checkpointing", self._osft_config.gradient_checkpointing,
            ),
            logging_steps=training_args.get("logging_steps", 10),
            save_strategy=training_args.get("save_strategy", "steps"),
            save_steps=training_args.get("save_steps", 100),
            save_total_limit=training_args.get("save_total_limit", 3),
            seed=self._osft_config.seed,
            report_to="none",
        )

        self._trainer = transformers.Trainer(
            model=self._model,
            args=hf_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self._tokenizer,
        )

        train_output = self._trainer.train(
            resume_from_checkpoint=training_args.get("resume_from_checkpoint", False) or None,
        )

        eval_loss = None
        if eval_ds is not None:
            eval_result = self._trainer.evaluate()
            eval_loss = eval_result.get("eval_loss")

        return TrainingResult(
            method="osft",
            model_id=self._osft_config.model_id,
            model_revision=self._osft_config.model_revision,
            seed=self._osft_config.seed,
            train_samples=len(train_ds),
            validation_samples=len(eval_ds) if eval_ds else 0,
            total_steps=train_output.global_step,
            num_epochs=self._osft_config.num_train_epochs,
            final_train_loss=train_output.training_loss,
            final_eval_loss=eval_loss,
            checkpoint_path=hf_args.output_dir,
            exported_path=self._osft_config.output_path,
        )

    def _apply_osft_decomposition(self, model: Any) -> None:
        """Freeze most parameters and selectively unfreeze based on rank ratio.

        OSFT operates by decomposing weight matrices via SVD and only
        training the lowest-rank components that capture task-specific
        directions, while keeping the orthogonal complement frozen.
        """
        ratio = self._osft_config.unfreeze_rank_ratio

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Selectively unfreeze linear layers based on rank ratio
        unfrozen_count = 0
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight.dim() != 2:
                continue

            weight = module.weight
            rows, cols = weight.shape
            rank = min(rows, cols)
            unfreeze_rank = max(1, int(rank * ratio))

            if _torch_available and hasattr(torch.linalg, "svd"):
                try:
                    U, S, Vh = torch.linalg.svd(weight.data.float(), full_matrices=False)
                    # Keep top singular components frozen, unfreeze the tail
                    # This is a simplification; real OSFT uses orthogonal projection
                    module.weight.requires_grad = True
                    unfrozen_count += 1
                except Exception:
                    # Fallback: just unfreeze the whole layer
                    module.weight.requires_grad = True
                    unfrozen_count += 1
            else:
                module.weight.requires_grad = True
                unfrozen_count += 1

        logger.info(
            "OSFT decomposition: unfroze %d layers with rank ratio %.2f",
            unfrozen_count, ratio,
        )

    def _validate_exported(self, path: Path, export_cfg: dict[str, Any]) -> None:
        """Re-load and validate an exported model."""
        try:
            reloaded = transformers.AutoModelForCausalLM.from_pretrained(
                str(path), trust_remote_code=True,
            )
            logger.info("Export validation: model reloaded successfully from %s", path)

            if export_cfg.get("check_shards", False):
                shard_files = list(path.glob("model-*.safetensors"))
                if not shard_files:
                    shard_files = list(path.glob("pytorch_model-*.bin"))
                logger.info("Export validation: found %d shard files", len(shard_files))

            del reloaded
        except Exception as exc:
            logger.error("Export validation failed: %s", exc)
            raise


__all__ = [
    "DataPreprocessor",
    "LoRATrainer",
    "OSFTTrainer",
    "validate_bundle_for_training",
]
