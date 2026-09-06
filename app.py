import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List

from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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


def build_rag_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()

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
# UI Logic
# =========================
def render_chat():
    history = get_history(SESSION_ID).messages

    for msg in history:
        template = user_template if msg.type == "human" else bot_template
        st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def handle_query(query: str):
    # Guardrail Check
    if not legal_guard.validate(query):
        st.error("⚠️ This question is out of scope. I only answer questions related to the Canadian Bill of Rights.")
        return
    
    config = {"configurable": {"session_id": SESSION_ID}}

    st.session_state.chain.invoke(
        {"input": query},
        config=config
    )

    render_chat()


# =========================
# Evals Logic
# =========================
def get_eval_chain(pdf_paths):
    docs = extract_documents(pdf_paths)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)
    retriever = vectorstore.as_retriever()
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

    # Auto-load persisted vector store on startup
    if st.session_state.chain is None and vectorstore_exists():
        vectorstore = load_vectorstore()
        st.session_state.chain = build_rag_chain(vectorstore)

    query = st.text_input("Ask a question, please")

    if query:
        if st.session_state.chain:
            handle_query(query)
        else:
            st.warning("Upload and process documents first.")

    with st.sidebar:
        st.subheader("Documents")
        files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Building knowledge base..."):
                docs = extract_documents(files)
                chunks = split_documents(docs)
                vectorstore = build_vectorstore(chunks)
                st.session_state.chain = build_rag_chain(vectorstore)
                st.success("Ready for answering questions.")


if __name__ == "__main__":
    main()