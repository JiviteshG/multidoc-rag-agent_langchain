import os
import sys
# Add the parent directory to path so we can import app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_eval_chain
from eval_dataset import eval_samples
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI

def run_evaluation():
    # 1. Point to your actual PDF files in the docs folder
    pdf_files = ["docs/constitution.pdf", "docs/canadian bill of rights.pdf"]
    
    # 2. Build the chain using your app's logic
    chain, retriever = get_eval_chain(pdf_files)
    
    results = []
    print("Starting evaluation...")

    for sample in eval_samples:
        config = {"configurable": {"session_id": "eval_session"}}
        
        # 1. Get the Answer from your LLM chain
        response_text = chain.invoke({"input": sample["question"]}, config=config)
        
        # 2. Get the actual PDF chunks (The "Contexts")
        # This is what Ragas needs to verify the answer isn't a hallucination
        retrieved_docs = retriever.invoke(sample["question"])
        contexts = [doc.page_content for doc in retrieved_docs]
        
        results.append({
            "question": sample["question"],
            "answer": response_text,
            "ground_truth": sample["ground_truth"],
            "contexts": contexts  # Real text from the PDFs
        })

    # 3. Run Ragas
    dataset = Dataset.from_list(results)
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    
    print("\n--- Evaluation Results ---")
    print(score)
    score.to_pandas().to_csv("evals/ragas_results.csv")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluation()