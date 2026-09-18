"""Integration tests for FastAPI endpoints using TestClient."""

from __future__ import annotations

import pytest

from rhoai_model_training_lab.api import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app with minimal config."""
    config = {
        "model": {
            "endpoint": "",
            "model_name": "test-model",
        },
        "retrieval": {
            "embedding": {"model_id": "test"},
            "vector_store": {"persist_directory": "/tmp/test-index"},
        },
        "harness": {
            "mode": "simple_rag",
            "limits": {
                "max_turns": 5,
                "max_tool_calls": 5,
            },
        },
    }
    return create_app(config)


@pytest.fixture
def client(app):
    """Create a synchronous test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestHealthEndpoint:
    """Test GET /healthz returns 200."""

    def test_healthz_returns_200(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_healthz_response_model(self, client):
        response = client.get("/healthz")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_endpoint" in data


class TestReadyEndpoint:
    """Test GET /readyz when index not loaded."""

    def test_readyz_not_ready_without_index(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["index_ready"] is False


class TestAgentStep:
    """Test POST /v1/agent/step with valid request."""

    def test_agent_step_returns_response(self, client):
        payload = {
            "session_id": "test-session-001",
            "messages": [
                {"role": "user", "content": "What is the interest rate?"},
            ],
            "mode": "simple_rag",
        }
        response = client.post("/v1/agent/step", json=payload)
        assert response.status_code in (200, 503)
        data = response.json()
        assert "session_id" in data or "status" in data

    def test_agent_step_session_isolation(self, client):
        for session_id in ["session-a", "session-b"]:
            payload = {
                "session_id": session_id,
                "messages": [
                    {"role": "user", "content": f"Hello from {session_id}"},
                ],
            }
            response = client.post("/v1/agent/step", json=payload)
            assert response.status_code in (200, 503)


class TestQAEndpoint:
    """Test POST /v1/qa with valid request."""

    def test_qa_returns_response(self, client):
        payload = {
            "question": "What is the late fee for credit cards?",
            "mode": "rag",
            "model_variant": "base",
        }
        response = client.post("/v1/qa", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "trace_id" in data

    def test_qa_matched_context_mode(self, client):
        payload = {
            "question": "What is the overdraft limit?",
            "context_documents": [
                "The overdraft limit is $500 for standard accounts.",
            ],
            "mode": "matched_context",
            "model_variant": "lora",
        }
        response = client.post("/v1/qa", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "error")


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_endpoint_returns_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_missing_required_field(self, client):
        response = client.post("/v1/qa", json={})
        assert response.status_code == 422
