"""Training configuration and result schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA fine-tuning configuration."""
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_revision: str = "main"
    tokenizer_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    train_file: str = ""
    validation_file: str = ""
    max_seq_length: int = 4096
    loss_masking: str = "assistant_only"

    output_dir: str = "checkpoints/lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    adapter_path: str = "models/lora-adapter"
    merged_path: str = "models/lora-merged"


class OSFTConfig(BaseModel):
    """OSFT fine-tuning configuration."""
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_revision: str = "main"
    tokenizer_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    unfreeze_rank_ratio: float = 0.25

    train_file: str = ""
    validation_file: str = ""
    max_seq_length: int = 4096
    loss_masking: str = "assistant_only"

    output_dir: str = "checkpoints/osft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    output_path: str = "models/osft-exported"


class TrainingResult(BaseModel):
    """Result from a training run."""
    method: str  # "lora" or "osft"
    model_id: str = ""
    model_revision: str = ""
    bundle_id: str = ""
    bundle_hash: str = ""
    canonical_hash: str = ""
    export_hash: str = ""
    seed: int = 42

    train_samples: int = 0
    validation_samples: int = 0
    total_steps: int = 0
    total_tokens: int = 0
    num_epochs: int = 0
    final_train_loss: float = 0.0
    final_eval_loss: float | None = None
    best_eval_loss: float | None = None
    best_checkpoint_step: int | None = None

    wall_time_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    gpu_name: str = ""

    checkpoint_path: str = ""
    adapter_path: str = ""
    merged_path: str = ""
    exported_path: str = ""

    mlflow_run_id: str = ""
    mlflow_experiment: str = ""
