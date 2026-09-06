import os
import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List

from pydantic import BaseModel, Field

from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from htmlTemplates import css, bot_template, user_template
from logic_guards.input_guards import LegalGuardrail

# Load env vars before any LangChain/OpenAI objects are instantiated
load_dotenv()

legal_guard = LegalGuardrail()

# =========================
# Configuration
# =========================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SESSION_ID = "default"
CHROMA_PERSIST_DIR = "./chroma_db"


# =========================
# Session Memory
# =========================
def get_history(session_id: str):
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}

    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.history_store[session_id]


# =========================
# Document Processing
# =========================
def extract_documents(files) -> List[Document]:
    docs = []
    for file in files:
        source = file.name if hasattr(file, "name") else os.path.basename(str(file))
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": source, "page": page_num + 1}
                ))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# =========================
# Vector Store
# =========================
def build_vectorstore(chunks: List[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)


def vectorstore_exists() -> bool:
    return os.path.exists(CHROMA_PERSIST_DIR) and bool(os.listdir(CHROMA_PERSIST_DIR))


def get_stored_document_names() -> List[str]:
    if not vectorstore_exists():
        return []
    try:
        vs = load_vectorstore()
        result = vs._collection.get(include=["metadatas"])
        sources = sorted(set(
            m.get("source", "") for m in result["metadatas"] if m.get("source")
        ))
        return sources
    except Exception:
        return []


# =========================
# RAG Chain (LCEL)
# =========================
def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_retriever(vectorstore):
    # Stage 1: fetch top-20 candidates via vector similarity (fast, coarse)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Stage 2: rerank those 20 with a cross-encoder, keep the top 5 (precise)
    # FlashRank runs locally — no API key required.
    # Cross-encoder reads query + chunk together, unlike embeddings which score them apart.
    compressor = FlashrankRerank(top_n=5)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_rag_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = build_retriever(vectorstore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer strictly using the provided context. "
            "At the end of your answer, on a new line, list the sources you used "
            "in this format — Sources: <filename>, Page <n>.\n\n{context}"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    ).with_config(run_name="multidoc-rag-query")


# =========================
# Structured Output
# =========================
class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question, grounded strictly in the provided context")
    sources: List[str] = Field(description="List of sources cited, formatted as 'filename.pdf, Page N'")


def build_structured_chain(vectorstore):
    """Single-shot chain that returns a typed RAGResponse instead of a raw string.

    Used by programmatic callers (API endpoints, evals) that need typed fields.
    Does NOT stream — JSON must be fully formed before it can be parsed.
    History is NOT managed here; callers own their own context window.
    """
    llm = ChatOpenAI().with_structured_output(RAGResponse)
    retriever = build_retriever(vectorstore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer strictly using the provided context. "
            "Return your response as a structured object with an 'answer' field "
            "and a 'sources' list (e.g. ['filename.pdf, Page 3']).\n\n{context}"
        )),
        ("human", "{input}"),
    ])

    return (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
    ).with_config(run_name="multidoc-rag-structured")


# =========================
# UI Logic
# =========================
def _format_bot_content(content: str) -> str:
    # Handles both "Source:" (singular) and "Sources:" (plural) from the LLM
    match = re.search(r'\n(Sources?:)', content)
    if match:
        answer = content[:match.start()].strip()
        sources_text = content[match.start(1):]
        return answer + f'<div class="sources-block">{sources_text}</div>'
    return content


def render_chat():
    history = get_history(SESSION_ID).messages

    for msg in history:
        if msg.type == "human":
            content = msg.content
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            content = _format_bot_content(msg.content)
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)


def stream_response(query: str):
    """Generator that yields text chunks from the chain stream.

    RunnableWithMessageHistory.stream() yields the inner chain's output chunks.
    Since the chain ends with StrOutputParser(), each chunk is a plain string.
    We guard against AIMessageChunk objects in case the parser is bypassed.
    History is auto-saved by RunnableWithMessageHistory once the stream is exhausted.
    """
    config = {"configurable": {"session_id": SESSION_ID}}
    for chunk in st.session_state.chain.stream({"input": query}, config=config):
        if isinstance(chunk, str):
            yield chunk
        elif hasattr(chunk, "content"):
            yield chunk.content


def handle_query(query: str):
    # Guardrail check
    if not legal_guard.validate(query):
        render_chat()
        st.write(user_template.replace("{{MSG}}", query), unsafe_allow_html=True)
        st.error("⚠️ This question is out of scope. I only answer questions related to the Canadian Bill of Rights.")
        return

    # Show existing history, then the new user message
    render_chat()
    st.write(user_template.replace("{{MSG}}", query), unsafe_allow_html=True)

    # Stream the bot response token-by-token.
    # st.write_stream() renders chunks as they arrive and returns the full text.
    # Retrieval (ChromaDB + reranker) runs first — blocking — then LLM streams.
    with st.spinner("Retrieving..."):
        gen = stream_response(query)
        first_chunk = next(gen, None)   # blocks until retrieval + first token ready

    if first_chunk is not None:
        def _full_stream():
            yield first_chunk
            yield from gen

        with st.chat_message("assistant"):
            st.write_stream(_full_stream())


# =========================
# Evals Logic
# =========================
def get_eval_chain(pdf_paths):
    docs = extract_documents(pdf_paths)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)
    retriever = build_retriever(vectorstore)
    return chain, retriever


# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title="MultiDoc RAG Agent", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    st.title("Multi-Document RAG Agent")

    if "chain" not in st.session_state:
        st.session_state.chain = None

    # Auto-load persisted vector store on startup (skip silently if store is corrupt)
    if st.session_state.chain is None and vectorstore_exists():
        try:
            vectorstore = load_vectorstore()
            st.session_state.chain = build_rag_chain(vectorstore)
        except Exception:
            pass

    # Populate knowledge base list once per session (refresh after processing)
    if "kb_documents" not in st.session_state:
        st.session_state.kb_documents = get_stored_document_names()

    with st.sidebar:
        st.subheader("Documents")

        # Knowledge base status box
        if st.session_state.kb_documents:
            with st.expander("📚 Knowledge Base", expanded=True):
                for doc in st.session_state.kb_documents:
                    st.markdown(f"✅ **{doc}**")
            st.markdown("---")

        files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if not files:
                st.warning("Please upload at least one PDF first.")
            else:
                with st.spinner("Building knowledge base..."):
                    docs = extract_documents(files)
                    chunks = split_documents(docs)
                    vectorstore = build_vectorstore(chunks)
                    st.session_state.chain = build_rag_chain(vectorstore)
                    st.session_state.kb_documents = get_stored_document_names()
                st.success(f"Ready! Processed {len(files)} file(s).")
                st.rerun()

    # Chat input (sticks to bottom, auto-clears after submit)
    query = st.chat_input("Ask a question...")

    if query:
        if st.session_state.chain:
            handle_query(query)
        else:
            render_chat()
            st.warning("Upload and process documents first.")
    else:
        render_chat()


if __name__ == "__main__":
    main()
