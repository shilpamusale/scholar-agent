# configs/settings.py

from pathlib import Path

# --- Project Root ---
# This computes the absolute path to the project's root directory
PROJECT_ROOT = Path(__file__).parent.parent

# --- Data Paths ---
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
VECTOR_STORE_PATH = str(
    PROCESSED_DATA_PATH / "chroma_db"
)  # Convert to string for ChromaDB
LOGS_PATH = PROJECT_ROOT / "logs"


# --- Data Ingestion ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Document Downloader ---
SEARCH_QUERY = "Anthropic dictionary learning interpretability sparse autoencoder"
MAX_RESULTS = 20

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- RAG Pipeline ---
# The number of documents to retrieve in the initial fast search
RETRIEVER_TOP_K = 15  # Reduced from 20

# The number of documents to pass to the final LLM after re-ranking
RERANKER_TOP_N = 3  # Reduced from 5

# --- Cross-Encoder Model for Re-ranking ---
CROSS_ENCODER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

# --- LLM ---
LLM_MODEL_NAME = "gemini-1.5-flash"
MAX_OUTPUT_TOKENS = 512  # New setting to control generation length

# --- External APIs ---
S2_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_FIELDS = "paperId,title,authors.name,year,references.paperId,references.title"
