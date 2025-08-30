from pathlib import Path

# --- PROJECT ROOT ---
# Define the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent

# --- PATHS ---
# Build paths relative to the project root
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
LOGS_PATH = PROJECT_ROOT / "logs"
VECTOR_STORE_PATH = PROCESSED_DATA_PATH / "chroma_db"

# --- DATA INGESTION ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- ARXIV DOWNLOADER ---
SEARCH_QUERY = "Anthropic dictionary learning interpretability sparse autoencoder"
MAX_RESULTS = 20

# --- RAG PIPELINE ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
LLM_MODEL_NAME = "gemini-1.5-flash"
RETRIEVER_TOP_K = 20
RERANKER_TOP_N = 5
