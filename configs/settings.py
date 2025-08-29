# configs/settings.py

from pathlib import Path

# --- Project Root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data and Log Paths ---
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"
LOGS_PATH = PROJECT_ROOT / "logs"

# --- RAG Pipeline Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Model Names ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
LLM_MODEL_NAME = "gemini-1.5-flash"

# --- ArXiv Search Settings ---
SEARCH_QUERY = "Anthropic dictionary learning interpretability sparse autoencoder"
MAX_RESULTS = 20
