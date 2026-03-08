import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ChromaDB
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
CHROMA_COLLECTION_NAME = "astro_knowledge"

# Memory
MAX_CONVERSATION_WINDOW = 10  # Keep last N turns (each turn = user + assistant)
MAX_SUMMARY_TOKENS = 500

# Retrieval
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score to include context
MAX_CONTEXT_TOKENS = 2000
TOP_K_RESULTS = 5

# LLM
LLM_TEMPERATURE = 0.7
LLM_MAX_RETRIES = 3

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
