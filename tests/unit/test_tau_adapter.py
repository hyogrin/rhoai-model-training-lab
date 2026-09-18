"""Tests for the τ-bench simulation adapter: filtering, validation, bridge."""

from __future__ import annotations

import json

import pytest

from rhoai_model_training_lab.tau_adapter import (
    filter_private_fields,
    validate_tool_call,
    TauAgentBridge,
    TauSimulationAdapter,
)


class TestFilterPrivateFields:
    """Test filter_private_fields removes hidden evaluator data."""

    def test_removes_expected_action(self):
        task = {
            "task_id": "task-001",
            "user_message": "Check my balance",
            "expected_action": "get_balance",
            "expected_actions": ["get_balance", "verify_identity"],
        }
        filtered = filter_private_fields(task)
        assert "expected_action" not in filtered
        assert "expected_actions" not in filtered
        assert filtered["task_id"] == "task-001"
        assert filtered["user_message"] == "Check my balance"

    def test_removes_hidden_goal(self):
        task = {
            "task_id": "task-002",
            "hidden_goal": "Transfer $500 from savings to checking",
        }
        filtered = filter_private_fields(task)
        assert "hidden_goal" not in filtered

    def test_removes_target_reward(self):
        task = {
            "scenario": "fee_waiver",
            "target_reward": 1.0,
        }
        filtered = filter_private_fields(task)
        assert "target_reward" not in filtered

    def test_removes_evaluator_state(self):
        task = {
            "state": "active",
            "evaluator_state": {"internal": True},
        }
        filtered = filter_private_fields(task)
        assert "evaluator_state" not in filtered

    def test_removes_golden_doc(self):
        task = {
            "question": "What is the fee?",
            "golden_doc": "doc_fees_001",
        }
        filtered = filter_private_fields(task)
        assert "golden_doc" not in filtered

    def test_removes_grading_rubric(self):
        task = {
            "task_id": "task-003",
            "grading_rubric": {"criteria": ["accuracy", "helpfulness"]},
        }
        filtered = filter_private_fields(task)
        assert "grading_rubric" not in filtered

    def test_removes_nested_private_fields(self):
        task = {
            "task_id": "task-004",
            "details": {
                "public_info": "visible",
                "expected_tool_calls": ["get_balance"],
                "nested": {
                    "private_instructions": "hidden",
                },
            },
        }
        filtered = filter_private_fields(task)
        assert "expected_tool_calls" not in filtered["details"]
        assert "private_instructions" not in filtered["details"]["nested"]
        assert filtered["details"]["public_info"] == "visible"

    def test_removes_in_lists(self):
        task = {
            "episodes": [
                {"task_id": "t1", "success_condition": "balance > 100"},
                {"task_id": "t2", "reward_criteria": "accuracy"},
            ],
        }
        filtered = filter_private_fields(task)
        assert "success_condition" not in filtered["episodes"][0]
        assert "reward_criteria" not in filtered["episodes"][1]

    def test_preserves_non_private(self):
        task = {
            "task_id": "task-005",
            "user_message": "Hello",
            "tools": [{"name": "get_balance"}],
        }
        filtered = filter_private_fields(task)
        assert filtered == task

    def test_returns_deep_copy(self):
        task = {"task_id": "task-006", "data": {"key": "value"}}
        filtered = filter_private_fields(task)
        filtered["data"]["key"] = "modified"
        assert task["data"]["key"] == "value"


class TestValidateToolCall:
    """Test validate_tool_call catches invalid arguments."""

    @pytest.fixture
    def balance_schema(self) -> dict:
        return {
            "function": {
                "name": "get_account_balance",
                "description": "Get the balance of a bank account",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                    },
                    "required": ["account_id"],
                },
            },
        }

    def test_valid_call(self, balance_schema):
        result = validate_tool_call(
            "get_account_balance",
            {"account_id": "ACC-123"},
            balance_schema,
        )
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_required_param(self, balance_schema):
        result = validate_tool_call(
            "get_account_balance",
            {},
            balance_schema,
        )
        assert result["valid"] is False
        assert any("Missing required" in e for e in result["errors"])

    def test_wrong_tool_name(self, balance_schema):
        result = validate_tool_call(
            "transfer_funds",
            {"account_id": "ACC-123"},
            balance_schema,
        )
        assert result["valid"] is False
        assert any("name mismatch" in e.lower() for e in result["errors"])

    def test_unknown_parameter(self, balance_schema):
        result = validate_tool_call(
            "get_account_balance",
            {"account_id": "ACC-123", "extra_param": "bad"},
            balance_schema,
        )
        assert result["valid"] is False
        assert any("Unknown parameter" in e for e in result["errors"])

    def test_wrong_type(self):
        schema = {
            "function": {
                "name": "set_limit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                    },
                    "required": ["limit"],
                },
            },
        }
        result = validate_tool_call("set_limit", {"limit": "not_a_number"}, schema)
        assert result["valid"] is False
        assert any("expected type" in e.lower() for e in result["errors"])

    def test_enum_violation(self):
        schema = {
            "function": {
                "name": "set_status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["active", "frozen"]},
                    },
                    "required": ["status"],
                },
            },
        }
        result = validate_tool_call("set_status", {"status": "deleted"}, schema)
        assert result["valid"] is False
        assert any("not in allowed values" in e for e in result["errors"])


class TestTauAgentBridge:
    """Test TauAgentBridge does not expose private fields to model."""

    def test_bridge_filters_messages(self):
        bridge = TauAgentBridge(backend_url="http://fake:8000")
        messages = [
            {
                "role": "system",
                "content": "System prompt",
                "expected_actions": ["action1"],
            },
            {
                "role": "user",
                "content": "User message",
                "hidden_goal": "secret",
            },
        ]

        filtered_msgs = []
        for msg in messages:
            filtered_msgs.append(filter_private_fields(msg))

        for msg in filtered_msgs:
            assert "expected_actions" not in msg
            assert "hidden_goal" not in msg

    def test_new_session_id_unique(self):
        bridge = TauAgentBridge()
        ids = [bridge.new_session_id() for _ in range(10)]
        assert len(set(ids)) == 10

    def test_reset_returns_session_id(self):
        bridge = TauAgentBridge()
        session_id = bridge.reset()
        assert session_id.startswith("tau_eval_")

    def test_reset_with_provided_id(self):
        bridge = TauAgentBridge()
        session_id = bridge.reset("custom-session")
        assert session_id == "custom-session"


class TestTauSimulationAdapter:
    """Test TauSimulationAdapter in stub mode."""

    def test_stub_mode_tool_execution(self):
        adapter = TauSimulationAdapter(domain="banking_knowledge")
        result = adapter.execute_tool("get_balance", {"account_id": "ACC-123"})
        assert result["status"] == "stub"
        assert "not connected" in result["error"].lower() or "not executed" in result["error"].lower()

    def test_reset_episode_stub(self):
        adapter = TauSimulationAdapter()
        result = adapter.reset_episode()
        assert result["status"] == "stub"

    def test_get_available_tools_empty_stub(self):
        adapter = TauSimulationAdapter()
        tools = adapter.get_available_tools()
        assert tools == []
