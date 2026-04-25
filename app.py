import streamlit as st
# from langchain import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS # vectore db stores locally instead of cloud
from langchain_openai import OpenAIEmbeddings

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

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() # create embeddings for text chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) # create vector store from text chunks and embeddings
    return vectorstore

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
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write("Files processed successfully!")
                st.write(vectorstore)


if __name__ == '__main__':
    main()