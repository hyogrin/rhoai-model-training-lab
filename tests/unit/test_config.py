"""Tests for config loading and environment variable expansion."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rhoai_model_training_lab.config import (
    expand_env_vars,
    load_yaml_config,
    load_training_config,
)


class TestLoadYamlConfig:
    """Test load_yaml_config with valid and invalid inputs."""

    def test_load_valid_file(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"key": "value", "nested": {"a": 1}}))
        result = load_yaml_config(cfg_file, expand_env=False)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_load_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml_config(missing)

    def test_load_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        result = load_yaml_config(empty, expand_env=False)
        assert result == {}

    def test_load_with_env_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_HOST", "example.com")
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text(yaml.dump({"host": "${TEST_HOST}", "port": 8080}))
        result = load_yaml_config(cfg_file, expand_env=True)
        assert result["host"] == "example.com"
        assert result["port"] == 8080

    def test_load_without_env_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_HOST", "example.com")
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text(yaml.dump({"host": "${TEST_HOST}"}))
        result = load_yaml_config(cfg_file, expand_env=False)
        assert result["host"] == "${TEST_HOST}"


class TestExpandEnvVars:
    """Test expand_env_vars replaces ${VAR} correctly."""

    def test_simple_replacement(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        assert expand_env_vars("${MY_VAR}") == "hello"

    def test_replacement_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = expand_env_vars("http://${HOST}:${PORT}/api")
        assert result == "http://localhost:8080/api"

    def test_missing_var_replaced_with_empty(self, monkeypatch):
        monkeypatch.delenv("UNLIKELY_MISSING_VAR_XYZ", raising=False)
        result = expand_env_vars("prefix-${UNLIKELY_MISSING_VAR_XYZ}-suffix")
        assert result == "prefix--suffix"

    def test_no_vars_unchanged(self):
        assert expand_env_vars("plain string") == "plain string"

    def test_dict_recursive(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "db.example.com")
        data = {"database": {"host": "${DB_HOST}", "port": 5432}}
        result = expand_env_vars(data)
        assert result["database"]["host"] == "db.example.com"
        assert result["database"]["port"] == 5432

    def test_list_recursive(self, monkeypatch):
        monkeypatch.setenv("ITEM", "expanded")
        data = ["${ITEM}", "plain"]
        result = expand_env_vars(data)
        assert result == ["expanded", "plain"]

    def test_non_string_passthrough(self):
        assert expand_env_vars(42) == 42
        assert expand_env_vars(3.14) == 3.14
        assert expand_env_vars(True) is True
        assert expand_env_vars(None) is None


class TestLoadTrainingConfig:
    """Test load_training_config validates profile names."""

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown training profile"):
            load_training_config("invalid_profile")

    def test_invalid_profile_not_in_valid(self):
        with pytest.raises(ValueError, match="Must be one of"):
            load_training_config("bert")

    def test_valid_profiles_accepted(self, tmp_path, monkeypatch):
        for profile in ("lora", "osft"):
            cfg_file = tmp_path / f"{profile}.yaml"
            cfg_file.write_text(yaml.dump({"model_id": "test"}))
            monkeypatch.setattr(
                "rhoai_model_training_lab.config.PROJECT_ROOT",
                tmp_path,
            )
            (tmp_path / "configs").mkdir(exist_ok=True)
            (tmp_path / "configs" / f"{profile}.yaml").write_text(
                yaml.dump({"model_id": "test"})
            )
            result = load_training_config(profile)
            assert result["model_id"] == "test"
