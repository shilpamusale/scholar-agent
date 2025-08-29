import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Make sure the src directory is in the path for imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_processing.ingestion import load_documents, split_documents, create_and_persist_vector_store

@patch('src.data_processing.ingestion.glob.glob')
@patch('src.data_processing.ingestion.PyPDFLoader')
def test_load_documents(mock_loader, mock_glob):
    """Tests that documents are loaded correctly."""
    mock_glob.return_value = ['dummy.pdf']
    mock_loader.return_value.load.return_value = [Document(page_content="test")]
    
    documents = load_documents('fake_path')
    assert len(documents) == 1
    assert documents[0].page_content == "test"
    mock_loader.assert_called_with('dummy.pdf')

def test_split_documents():
    """Tests that documents are split into chunks."""
    docs = [Document(page_content="a" * 2000)]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert len(chunks[0].page_content) <= 1000

@patch('src.data_processing.ingestion.Chroma.from_documents')
@patch('src.data_processing.ingestion.SentenceTransformerEmbeddings')
def test_create_vector_store(mock_embeddings, mock_chroma):
    """Tests that the vector store creation is called."""
    chunks = [Document(page_content="test chunk")]
    create_and_persist_vector_store(chunks)
    mock_embeddings.assert_called_once()
    mock_chroma.assert_called_once()
