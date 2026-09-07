"""
FastAPI endpoint tests.
The RAG chain and vectorstore are mocked — no LLM calls, no ChromaDB needed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app import RAGResponse


@pytest.fixture
def client():
    """TestClient with chain and vectorstore mocked out."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = RAGResponse(
        answer="The Canadian Bill of Rights protects freedom of speech.",
        sources=["canadian_bill_of_rights.pdf, Page 2"],
    )

    with patch("api.vectorstore_exists", return_value=True), \
         patch("api.load_vectorstore", return_value=MagicMock()), \
         patch("api.build_structured_chain", return_value=mock_chain), \
         patch("api.guardrail") as mock_guard:

        mock_guard.validate.return_value = True

        from api import app
        with TestClient(app) as c:
            yield c, mock_chain, mock_guard


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    c, _, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── /query ───────────────────────────────────────────────────────────────────

def test_query_returns_rag_response(client):
    c, mock_chain, _ = client
    response = c.post("/query", json={"query": "What rights are protected?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_query_calls_chain_with_input(client):
    c, mock_chain, _ = client
    c.post("/query", json={"query": "What is section 1?"})
    mock_chain.ainvoke.assert_called_once_with({"input": "What is section 1?"})


def test_query_out_of_scope_returns_400(client):
    c, _, mock_guard = client
    mock_guard.validate.return_value = False
    response = c.post("/query", json={"query": "What is the weather in Toronto?"})
    assert response.status_code == 400
    assert "out of scope" in response.json()["detail"].lower()


def test_query_missing_body_returns_422(client):
    c, _, _ = client
    response = c.post("/query", json={})
    assert response.status_code == 422


def test_query_empty_string_passes_to_chain(client):
    c, mock_chain, _ = client
    c.post("/query", json={"query": ""})
    mock_chain.ainvoke.assert_called_once_with({"input": ""})
