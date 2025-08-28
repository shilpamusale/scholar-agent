# configs/settings.py

# --- Data Paths ---
RAW_DATA_PATH = "data/raw"
VECTOR_STORE_PATH = "data/processed/chroma_db"

# --- Data Ingestion Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Downloader Settings ---
SEARCH_QUERY = "Anthropic dictionary learning interpretability sparse autoencoder"
NUM_PAPERS_TO_DOWNLOAD = 20