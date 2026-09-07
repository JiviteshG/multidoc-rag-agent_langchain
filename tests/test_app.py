"""
Unit tests for pure functions in app.py.
No LLM calls — these run fast with no API key required.
"""
import pytest
from unittest.mock import patch
from langchain_core.documents import Document

# Patch streamlit session_state so app.py can be imported outside a Streamlit context
with patch("streamlit.session_state", {}):
    from app import split_documents, _format_bot_content, RAGResponse, vectorstore_exists


# ── split_documents ──────────────────────────────────────────────────────────

class TestSplitDocuments:
    def test_long_document_is_chunked(self):
        # CharacterTextSplitter splits on "\n" — content must have newlines to be chunked
        doc = Document(page_content=("word " * 200 + "\n") * 10, metadata={"source": "test.pdf", "page": 1})
        chunks = split_documents([doc])
        assert len(chunks) > 1

    def test_short_document_stays_as_one_chunk(self):
        doc = Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 1})
        chunks = split_documents([doc])
        assert len(chunks) == 1

    def test_metadata_is_preserved_in_chunks(self):
        doc = Document(page_content="word " * 600, metadata={"source": "bill.pdf", "page": 3})
        chunks = split_documents([doc])
        assert all(c.metadata["source"] == "bill.pdf" for c in chunks)

    def test_multiple_documents_all_chunked(self):
        docs = [
            Document(page_content="word " * 600, metadata={"source": f"doc{i}.pdf", "page": 1})
            for i in range(3)
        ]
        chunks = split_documents(docs)
        assert len(chunks) >= 3


# ── _format_bot_content ──────────────────────────────────────────────────────

class TestFormatBotContent:
    def test_sources_section_wrapped_in_div(self):
        content = "The answer.\n\nSources: bill.pdf, Page 2"
        result = _format_bot_content(content)
        assert '<div class="sources-block">' in result

    def test_answer_text_preserved(self):
        content = "The answer.\n\nSources: bill.pdf, Page 2"
        result = _format_bot_content(content)
        assert "The answer." in result

    def test_singular_source_also_matched(self):
        content = "The answer.\n\nSource: bill.pdf, Page 1"
        result = _format_bot_content(content)
        assert '<div class="sources-block">' in result

    def test_no_sources_returns_content_unchanged(self):
        content = "Just an answer with no citation."
        result = _format_bot_content(content)
        assert result == content


# ── RAGResponse ──────────────────────────────────────────────────────────────

class TestRAGResponse:
    def test_valid_response_constructed(self):
        r = RAGResponse(answer="Yes.", sources=["bill.pdf, Page 1"])
        assert r.answer == "Yes."
        assert r.sources == ["bill.pdf, Page 1"]

    def test_multiple_sources_accepted(self):
        r = RAGResponse(answer="Yes.", sources=["a.pdf, Page 1", "b.pdf, Page 3"])
        assert len(r.sources) == 2

    def test_empty_sources_list_accepted(self):
        r = RAGResponse(answer="No relevant sources.", sources=[])
        assert r.sources == []

    def test_missing_answer_raises(self):
        with pytest.raises(Exception):
            RAGResponse(sources=["bill.pdf, Page 1"])

    def test_missing_sources_raises(self):
        with pytest.raises(Exception):
            RAGResponse(answer="Yes.")

    def test_serialises_to_dict(self):
        r = RAGResponse(answer="Yes.", sources=["bill.pdf, Page 1"])
        d = r.model_dump()
        assert d == {"answer": "Yes.", "sources": ["bill.pdf, Page 1"]}


# ── vectorstore_exists ───────────────────────────────────────────────────────

class TestVectorstoreExists:
    def test_returns_false_when_directory_missing(self, tmp_path):
        with patch("app.CHROMA_PERSIST_DIR", str(tmp_path / "nonexistent")):
            assert vectorstore_exists() is False

    def test_returns_false_when_directory_empty(self, tmp_path):
        empty_dir = tmp_path / "chroma"
        empty_dir.mkdir()
        with patch("app.CHROMA_PERSIST_DIR", str(empty_dir)):
            assert vectorstore_exists() is False

    def test_returns_true_when_directory_has_files(self, tmp_path):
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        (chroma_dir / "chroma.sqlite3").write_text("data")
        with patch("app.CHROMA_PERSIST_DIR", str(chroma_dir)):
            assert vectorstore_exists() is True
