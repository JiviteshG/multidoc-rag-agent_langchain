import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import get_eval_chain
from eval_dataset import eval_samples
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def run_evaluation():
    # PDF files in the docs folder
    pdf_files = ["docs/constitution.pdf", "docs/canadian bill of rights.pdf"]
    
    # Build the chain using your app's logic
    chain, retriever = get_eval_chain(pdf_files)
    
    results = []
    print("Starting evaluation...")

    for sample in eval_samples:
        config = {"configurable": {"session_id": "eval_session"}}
        
        # Get the Answer from your LLM chain
        response_text = chain.invoke({"input": sample["question"]}, config=config)
        
        # Get the actual PDF chunks (The "Contexts")
        # This is what Ragas needs to verify the answer isn't a hallucination
        retrieved_docs = retriever.invoke(sample["question"])
        contexts = [doc.page_content for doc in retrieved_docs]
        
        results.append({
            "question": sample["question"],
            "answer": response_text,
            "ground_truth": sample["ground_truth"],
            "contexts": contexts  # Real text from the PDFs
        })

    # Run Ragas
    dataset = Dataset.from_list(results)
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        embeddings=OpenAIEmbeddings() 
    )
    # Store results in a CSV for analysis
    print("\n--- Evaluation Results ---")
    print(score)
    score.to_pandas().to_csv("evals/ragas_results.csv")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluation()