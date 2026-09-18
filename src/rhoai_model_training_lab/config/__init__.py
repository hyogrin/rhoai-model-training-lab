"""Centralized configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Return the project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def load_env(env_path: Path | None = None) -> None:
    """Load .env file from project root or specified path."""
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} references in config values."""
    if isinstance(value, str):
        if "${" in value:
            result = value
            for match_start in range(len(value)):
                if value[match_start : match_start + 2] == "${":
                    end = value.find("}", match_start)
                    if end != -1:
                        var_name = value[match_start + 2 : end]
                        env_val = os.environ.get(var_name, "")
                        result = result.replace(f"${{{var_name}}}", env_val)
            return result
        return value
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_yaml_config(path: str | Path, expand_env: bool = True) -> dict[str, Any]:
    """Load a YAML configuration file with optional env var expansion.

    Args:
        path: Path to the YAML file (absolute or relative to project root).
        expand_env: Whether to expand ${VAR} references.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    if expand_env:
        data = expand_env_vars(data)

    return data


def load_bundle_config(release_config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the data-release configuration for bundle operations."""
    if release_config_path is None:
        release_config_path = "configs/data-release.yaml"
    return load_yaml_config(release_config_path)


def load_training_config(profile: str) -> dict[str, Any]:
    """Load a training profile configuration.

    Args:
        profile: One of 'lora' or 'osft'.
    """
    valid_profiles = ("lora", "osft")
    if profile not in valid_profiles:
        raise ValueError(f"Unknown training profile: {profile}. Must be one of {valid_profiles}")
    return load_yaml_config(f"configs/{profile}.yaml")


def load_eval_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load evaluation configuration."""
    if path is None:
        path = "configs/eval.yaml"
    return load_yaml_config(path)


def load_rag_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load RAG harness configuration."""
    if path is None:
        path = "configs/rag.yaml"
    return load_yaml_config(path)
