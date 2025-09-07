# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
ingestion.py: Core data processing pipeline for the RAG vector store.

This module is responsible for the end-to-end process of taking raw PDF documents
and converting them into a persistent, searchable ChromaDB vector store. This
vector store serves as the primary knowledge base for the Retrieval-Augmented
Generation (RAG) capabilities of the Scholar-Agent.

The pipeline consists of three main stages:
1.  Loading: Scans a directory for PDF files and loads their text content.
2.  Splitting: Chunks the document text into manageable, overlapping segments.
3.  Embedding & Storing: Converts text chunks into vector embeddings using a
    SentenceTransformer model and persists them to a ChromaDB database.

This script is designed to be called by an entrypoint (e.g., scripts/ingest.py)
or as part of a larger automated workflow.
"""

# src/data_processing/ingestion.py

import glob
import os
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "ingestion_pipeline")


def load_documents(path: str) -> list[dict]:
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
            logger.info(f"Successfully loaded {os.path.basename(pdf_file)}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file}. Error: {e}")
    logger.info(f"Total document pages loaded: {len(documents)}")
    return documents


def split_documents(documents: list[dict]) -> list[dict]:
    """Splits the loaded documents into smaller chunks."""
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from documents.")
    return chunks


def create_and_persist_vector_store(chunks: list[dict]):
    """Creates and persists a ChromaDB vector store from document chunks."""
    logger.info("Creating vector store...")
    embedding_model = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    # Ensure we clear out the old directory to avoid stale data
    if os.path.exists(settings.VECTOR_STORE_PATH):
        logger.info(f"Removing old vector store at {settings.VECTOR_STORE_PATH}")
        shutil.rmtree(settings.VECTOR_STORE_PATH)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(settings.VECTOR_STORE_PATH),
        )
        logger.info(f"Vector store created and persisted at {settings.VECTOR_STORE_PATH}")
    except Exception as e:
        logger.error(f"Failed to create vector store. Error: {e}")


def run_ingestion_pipeline():
    """The main function to run the entire data ingestion and processing pipeline."""
    logger.info("Starting data ingestion pipeline...")
    documents = load_documents(settings.RAW_DATA_PATH)
    if documents:
        chunks = split_documents(documents)
        create_and_persist_vector_store(chunks)
        logger.info("Data ingestion pipeline finished successfully.")
    else:
        logger.warning("No documents were loaded, skipping pipeline.")
