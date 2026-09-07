import os
import pytest

# Set dummy API keys before any imports so LangChain/OpenAI don't fail at import time
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
