# scripts/ingest.py

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_processing.ingestion import run_ingestion_pipeline  # noqa: E402

if __name__ == "__main__":
    run_ingestion_pipeline()
