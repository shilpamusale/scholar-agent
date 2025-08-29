# src/data_processing/ingestion.py

import glob
import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Import configurations and utilities
import configs.settings as settings
from src.utils.logging_config import setup_logging

# Setup logger
logger = setup_logging(__name__, "ingestion_module")


def load_documents(path: str) -> List[dict]:
    """Loads all PDF documents from a given directory path."""
    logger.info(f"Loading documents from {path}...")
    pdf_files = glob.glob(os.path.join(path, "*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in the specified directory.")
        return []

    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load or process {pdf_file}. Error: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: List[dict]) -> List[dict]:
    """Splits the loaded documents into smaller chunks."""
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


def create_and_persist_vector_store(chunks: List[dict]):
    """Creates and persists a ChromaDB vector store from document chunks."""
    logger.info("Creating vector store...")
    embedding_model = SentenceTransformerEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=settings.VECTOR_STORE_PATH,
        )
        logger.info(
            f"Vector store created and persisted at {settings.VECTOR_STORE_PATH}"
        )
    except Exception as e:
        logger.error(f"Failed to create vector store. Error: {e}")


def run_ingestion_pipeline():
    """Runs the full data ingestion pipeline."""
    logger.info("Starting data ingestion pipeline...")
    documents = load_documents(settings.RAW_DATA_PATH)
    if documents:
        chunks = split_documents(documents)
        create_and_persist_vector_store(chunks)
        logger.info("Data ingestion pipeline finished successfully.")
    else:
        logger.warning("No documents were loaded. Aborting pipeline.")
