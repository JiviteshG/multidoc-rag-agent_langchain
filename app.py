import streamlit as st
# from langchain import OpenAI
from dotenv import load_dotenv
import os

def main():
    load_dotenv()  # Load environment variables from .env file
    st.set_page_config(page_title="MultiDoc RAG Agent", page_icon=":books:")

    st.header("MultiDoc RAG Agent with LangChain and Streamlit :books:")
    st.title("MultiDoc RAG Agent")

    st.text_input("Enter your question here:", key="user_input")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your documents here and click on Process", type=["pdf", "txt"], key="file_uploader")
        st.button("Process") #, on_click=process_documents)




if __name__ == '__main__':
    main()