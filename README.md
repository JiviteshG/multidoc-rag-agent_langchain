# Multi-Doc RAG Agent with LangChain 📚🤖

A Retrieval-Augmented Generation (RAG) conversational application that allows you to upload and chat with multiple PDF documents simultaneously. Built using LangChain and Streamlit.

## 🌟 Features
- **Multi-PDF Processing:** Upload multiple PDF documents at once and query across all of them.
- **Conversational Memory:** The agent remembers the context of the conversation, allowing for natural, continuous follow-up questions.
- **RAG Architecture:** Leverages semantic search and LLMs to fetch the most relevant text chunks from your documents to generate accurate answers.
- **Interactive Web UI:** Features a clean and responsive chat interface powered by Streamlit.

## 🛠️ Technologies Used
- **[LangChain](https://python.langchain.com/):** Core framework for orchestrating LLMs, vector stores, and conversational chains.
- **[Streamlit](https://streamlit.io/):** For building the front-end chat interface.
- **Embeddings & Vector Store:** Used to convert text into vector embeddings and store them for fast semantic search (typically FAISS or ChromaDB).
- **LLM Integration:** Compatible with models like OpenAI GPT or HuggingFace alternatives.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- An API Key for your chosen LLM (e.g., OpenAI API Key or HuggingFace Token)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/JiviteshG/multidoc-rag-agent_langchain.git](https://github.com/JiviteshG/multidoc-rag-agent_langchain.git)
   cd multidoc-rag-agent_langchain
    ```
   
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate  
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Set up Environment Variables:**
Create a .env file in the root of the project and securely store your API keys:
   ```
   Code snippet
   OPENAI_API_KEY=your_openai_api_key_here
   # HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here (if applicable)
   ```
  
**💻 Usage**
1. Start the application by running:
   ```bash
   streamlit run app.py

2. The application will launch in your default web browser.

3. Use the sidebar to Upload your PDF documents.

4. Click on Process to chunk the text and build the vector database.

5. Once processing is complete, use the text input at the bottom to start asking questions about your documents!

📁 Repository Structure

📦 multidoc-rag-agent_langchain/

├── app.py               # Main Streamlit application and logic

├── htmlTemplates.py     # HTML/CSS UI components for user & bot messages

├── requirements.txt     # List of project dependencies

├── README.md            # Project documentation

├── .gitignore           # Ignored files for version control

└── .python-version      # Specifies the Python version used

├── evals/

│   ├── eval_dataset.py       # Your golden questions

│   ├── run_ragas_eval.py     # The execution Ragas script

│   └── ragas_report_v1.csv   # The results of evals 

## 📊 Performance Evaluation
The system was evaluated using the **Ragas** framework against a "Golden Dataset" of legal questions derived from the Canadian Constitution and Bill of Rights.

![RAG Evaluation Metrics](evals\eval_results_plot.png)

### Key Results:
- **Faithfulness (1.00):** The model demonstrated zero hallucinations, with all answers being 100% supported by the source documents.
- **Answer Relevancy (0.96):** The model provides highly relevant answers that directly address the user's query.
### 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

### 📜 License
This project is open-source and available under the MIT License.
