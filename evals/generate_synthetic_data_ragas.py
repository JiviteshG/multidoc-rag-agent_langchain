##################################################
## UNDER DEVELOPMENT - DO NOT USE THIS FILE YET
## Ragas v0.3+ Factory-based Testset Generation Script
## Unable to run until Ragas v0.3+ is released with the new factory-based API and fixes for the JSON parsing and Client errors.
##################################################
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# v0.3+ Factory Imports
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

def generate_testset():
    load_dotenv()
    
    # Initialize the OpenAI client
    client = OpenAI()

    print("1. Loading Document...")
    loader = PyPDFLoader("docs/canadian bill of rights.pdf")
    documents = loader.load()

    print("2. Pre-chunking with larger context...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)

    print("3. Initializing Models (v0.3 Client Syntax)...")
    # Pass the client instance to both factories
    generator_llm = llm_factory("gpt-4o-mini", client=client)
    generator_embeddings = embedding_factory("openai", client=client)

    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings
    )

    print("⏳ 4. Generating Synthetic Questions...")
    # Bypass the previous JSON parsing error and the Client error
    dataset = generator.generate_with_langchain_docs(
        chunked_docs, 
        testset_size=10
    )

    print("5. Saving to CSV...")
    test_df = dataset.to_pandas()
    os.makedirs("evals", exist_ok=True)
    test_df.to_csv("evals/synthetic_eval_dataset.csv", index=False)
    
    print("✅ Success! File saved at evals/synthetic_eval_dataset.csv")

if __name__ == "__main__":
    generate_testset()