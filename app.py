import streamlit as st
# from langchain import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # throusand characters per chunk
        chunk_overlap=200, # protect against cutting off important context
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def main():
    load_dotenv()  # Load environment variables from .env file
    st.set_page_config(page_title="MultiDoc RAG Agent", page_icon=":books:")

    st.header("MultiDoc RAG Agent with LangChain and Streamlit :books:")
    st.title("MultiDoc RAG Agent")

    st.text_input("Enter your question here:", key="user_input")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents here and click on Process", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing files..."):
                raw_text = get_pdf_text(pdf_docs)
                
                # get text chunks
                text_chunks =  get_text_chunks(raw_text)
                st.write(text_chunks)
                # create vector store



if __name__ == '__main__':
    main()