# tests/test_rag_pipeline.py

import os

# Make sure the src directory is in the path for imports
import sys
from unittest.mock import MagicMock, patch

from langchain_core.runnables import Runnable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.rag_pipeline.core import create_rag_chain


@patch("src.rag_pipeline.core.ChatGoogleGenerativeAI")
@patch("src.rag_pipeline.core.SentenceTransformerEmbeddings")
@patch("src.rag_pipeline.core.Chroma")
def test_create_rag_chain(mock_chroma, mock_embeddings, mock_llm):
    """
    Tests that the RAG chain is created successfully and is a runnable object.
    This test mocks all external dependencies.
    """
    # Configure mocks to return mock objects
    mock_embeddings.return_value = MagicMock()
    mock_chroma.return_value.as_retriever.return_value = MagicMock()
    mock_llm.return_value = MagicMock()

    # Create the RAG chain
    rag_chain = create_rag_chain()

    # Assert that the chain is a valid, runnable LangChain object
    assert rag_chain is not None
    assert isinstance(rag_chain, Runnable)

    # Assert that our external services were initialized
    mock_chroma.assert_called_once()
    mock_embeddings.assert_called_once()
    mock_llm.assert_called_once()
