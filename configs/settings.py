# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
settings.py: Central configuration for the Scholar-Agent project.

This module serves as the single source of truth for all static, non-secret
configuration parameters used throughout the application. By centralizing these
settings, we avoid hardcoding "magic numbers" in the application logic, making
the project easier to maintain, configure, and debug.

This file includes:
- File system paths for data, logs, and other artifacts.
- Parameters for data ingestion and processing (e.g., chunk size).
- Model names for embeddings, cross-encoders, and LLMs.
- Hyperparameters for the RAG and agentic pipelines.
- Base URLs and parameters for external APIs.

Secrets and credentials (e.g., API keys) are NOT stored here. They should be
managed via environment variables and loaded from a .env file.
"""

# configs/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# --- Project Root ---
# This computes the absolute path to the project's root directory
PROJECT_ROOT = Path(__file__).parent.parent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file.")


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

# The standard, cost-effective model for general tasks like routing and simple generation.
# This model is optimized for speed and efficiency.
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

# The advanced, powerful model for complex, high-stakes reasoning tasks.
# This is used for the critical Text-to-Cypher generation where accuracy is paramount.
LLM_MODEL_NAME_ADVANCED = "gemini-1.5-pro-latest"
# The maximum number of tokens to generate in a single response from the LLMs.
MAX_OUTPUT_TOKENS = 2048

# --- External APIs ---
S2_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_FIELDS = "paperId,title,authors.name,year,references.paperId,references.title"
