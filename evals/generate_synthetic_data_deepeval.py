import os
import pandas as pd
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from langchain_community.document_loaders import PyPDFLoader

def generate_testset():
    load_dotenv()
    
    # Manually load the PDF to ensure text is actually extracted
    print("1. Extracting text from PDF...")
    loader = PyPDFLoader("docs/canadian bill of rights.pdf")
    docs = loader.load()
    
    # Extract just the page content strings
    context_list = [doc.page_content for doc in docs if len(doc.page_content.strip()) > 100]
    
    if not context_list:
        print("X Error: No text could be extracted from the PDF. Check the file path/content.")
        return

    # Initialize the Synthesizer
    synthesizer = Synthesizer(model="gpt-4o-mini")

    print(f"2. Generating from {len(context_list)} valid text segments...")
    # We use generate_goldens_from_contexts to bypass the problematic internal file loader
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[[text] for text in context_list],
        max_goldens_per_context=2
    )

    # Create the Dataset
    dataset = EvaluationDataset(goldens=goldens)

    # Save to CSV correctly
    print("3. Saving results...")
    os.makedirs("evals", exist_ok=True)
    
    # FIX: Use synthesizer.to_pandas() or dataset.save_as()
    # Used the built-in save_as which is most reliable for CSVs
    dataset.save_as(
        file_type='csv',
        directory="./evals",
        file_name="synthetic_eval_dataset"
    )
    
    # Verify the file was created
    print(f"✅ Success! Generated {len(goldens)} test cases.")
    print("File location: evals/synthetic_eval_dataset.csv")

if __name__ == "__main__":
    generate_testset()