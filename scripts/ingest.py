# scripts/ingest.py

import glob
import os
import sys
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

import configs.settings as settings

# Import configurations and utilities
from src.utils.logging_config import setup_logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


# Setup logger
logger = setup_logging(__name__, "ingestion")


def load_documents(path: str) -> List[dict]:
    """
    Loads all PDF documents from a given directory path.

    Args:
        path (str): The path to the directory containing PDF files.

    Returns:
        List[dict]: A list of loaded document objects.
    """
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
            logger.info(f"Successfully loaded {pdf_file}")
        except Exception as e:
            logger.error(f"Failed to load or process {pdf_file}. Error: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: List[dict]) -> List[dict]:
    """
    Splits the loaded documents into smaller chunks for processing.

    Args:
        documents (List[dict]): A list of loaded document objects.

    Returns:
        List[dict]: A list of text chunks (documents).
    """
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks: List[dict]):
    """
    Creates and persists a ChromaDB vector store from the document chunks.

    Args:
        chunks (List[dict]): A list of text chunks.
    """
    logger.info("Creating vector store...")

    embedding_model = SentenceTransformerEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

    try:
        # vector_store = Chroma.from_documents(
        #     documents=chunks,
        #     embedding=embedding_model,
        #     persist_directory=settings.VECTOR_STORE_PATH,
        # )
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


if __name__ == "__main__":
    logger.info("Starting data ingestion pipeline...")

    # 1. Load documents from the raw data directory
    documents = load_documents(settings.RAW_DATA_PATH)

    if documents:
        # 2. Split the documents into chunks
        chunks = split_documents(documents)

        # 3. Create and persist the vector store
        create_vector_store(chunks)

        logger.info("Data ingestion pipeline finished successfully.")
    else:
        logger.warning("No documents were loaded. Aborting pipeline.")
