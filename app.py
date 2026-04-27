import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from htmlTemplates import css, bot_template, user_template


# =========================
# Configuration
# =========================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SESSION_ID = "default"


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
def extract_text(files) -> str:
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def split_text(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


# =========================
# Vector Store
# =========================
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


# =========================
# RAG Chain (LCEL)
# =========================
def build_rag_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly using the provided context:\n\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
    )


# =========================
# UI Logic
# =========================
def render_chat():
    history = get_history(SESSION_ID).messages

    for msg in history:
        template = user_template if msg.type == "human" else bot_template
        st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def handle_query(query: str):
    config = {"configurable": {"session_id": SESSION_ID}}

    st.session_state.chain.invoke(
        {"input": query},
        config=config
    )

    render_chat()


# =========================
# Main App
# =========================
def main():
    load_dotenv()

    st.set_page_config(page_title="MultiDoc RAG Agent", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    st.title("Multi-Document RAG Agent")

    if "chain" not in st.session_state:
        st.session_state.chain = None

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
                text = extract_text(files)
                chunks = split_text(text)
                vectorstore = build_vectorstore(chunks)

                st.session_state.chain = build_rag_chain(vectorstore)

                st.success("Ready for answering questions.")


if __name__ == "__main__":
    main()