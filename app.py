import streamlit as st
import torch
# from langchain import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS # vectore db stores locally instead of cloud
from langchain_openai import OpenAIEmbeddings # TO be removed
# from langchain_community.embeddings import HuggingFaceEmbeddings
# to fix langchain.memory import ConversationBufferMemory decication used pip install "langchain==0.3.27" 
# #https://stackoverflow.com/questions/76663833/langchain-memory-is-not-recognised-in-my-import-statement
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI   
from htmlTemplates import css, bot_template, user_template
import langchain
from langchain.globals import set_verbose 

set_verbose(True)

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
    # Use Qwen3 embedding model from HuggingFaceEmbeddings instead of OpenAIEmbeddings
    # model_name = "Qwen/Qwen3-Embedding-0.6B"

    # model_kwargs = {
    #     "device": "cuda"
    #         if torch.cuda.is_available() else "cpu" # Use GPU if available
    # } 

    # encode_kwargs = {  
    #     "normalize_embeddings": True,  # Normalize embeddings for better performance
    # }

    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) # create vector store from text chunks and embeddings
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()  # language model to use for generating responses

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(), # use the vector store as a retriever for relevant chunks
        memory=memory # use conversational memory to keep track of chat history
    )
    return conversation_chain


def main():
    load_dotenv()  # Load environment variables from .env file
    st.set_page_config(page_title="MultiDoc RAG Agent", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    # Initialize conversation in session state if it doesn't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("MultiDoc RAG Agent with LangChain and Streamlit :books:")
    st.title("MultiDoc RAG Agent")

    st.text_input("Enter your question here:", key="user_input")

    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

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
                
                # Create a conversation chain with the vector store and a language model
                # conversation is a variable that will hold the conversation chain for the user session using session state
                # you can access the conversation chain later in the app using st.session_state.conversation outside of the sidebar
                # session state allows you to "persist" data across user interactions in Streamlit, so the conversation chain will be available as long as the user session is active
                st.session_state.conversation = get_conversation_chain(vectorstore) 





if __name__ == '__main__':
    main()