"""
FastAPI layer for the MultiDoc RAG agent.

Exposes the same RAG pipeline as the Streamlit app but as a REST API.
Uses build_structured_chain() (typed RAGResponse output) instead of the
streaming chain, so callers get machine-readable JSON with typed fields.

Run:
    uvicorn api:app --reload --port 8000

Docs available at http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from app import (
    RAGResponse,
    build_structured_chain,
    load_vectorstore,
    vectorstore_exists,
)
from logic_guards.input_guards import LegalGuardrail

guardrail = LegalGuardrail()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load vectorstore and build chain once on startup — not per request.
    # Requires documents to have been processed via the Streamlit app first.
    if not vectorstore_exists():
        raise RuntimeError(
            "No knowledge base found. Upload and process PDFs via the Streamlit app first."
        )
    vectorstore = load_vectorstore()
    app.state.chain = build_structured_chain(vectorstore)
    yield


app = FastAPI(
    title="MultiDoc RAG API",
    description=(
        "Query the RAG knowledge base over uploaded documents. "
        "Returns a typed response with answer and source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=RAGResponse)
async def query_endpoint(request: QueryRequest):
    """Submit a question and receive a structured answer with source citations."""
    if not guardrail.validate(request.query):
        raise HTTPException(
            status_code=400,
            detail="Query is out of scope. This API only answers questions about the Canadian Bill of Rights.",
        )
    result = await app.state.chain.ainvoke({"input": request.query})
    return result
