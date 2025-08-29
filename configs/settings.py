# configs/settings.py
import os

# --- Project Root ---
# This gives us an absolute path to the project's root directory
# All other paths will be built from this
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Data Paths ---
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw")
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data/processed/chroma_db")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")


# --- Data Ingestion Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- LLM Model ---
LLM_MODEL_NAME = "gemini-1.5-flash"

# --- Downloader Settings ---
SEARCH_QUERY = "Anthropic dictionary learning interpretability sparse autoencoder"
NUM_PAPERS_TO_DOWNLOAD = 20
