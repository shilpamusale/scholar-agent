import os
import sys
from unittest.mock import MagicMock, patch

from langchain_core.runnables.base import RunnableSequence

# Add the project root to the Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# This import should come after the path modification
from src.rag_pipeline.core import create_rag_chain  # noqa: E402


@patch("src.rag_pipeline.core.Ranker")
@patch("src.rag_pipeline.core.ChatGoogleGenerativeAI")
@patch("src.rag_pipeline.core.SentenceTransformerEmbeddings")
@patch("src.rag_pipeline.core.Chroma")
def test_create_rag_chain(mock_chroma, mock_embeddings, mock_llm, mock_ranker):
    """
    Tests that the RAG chain is created successfully and is a runnable object.
    This test mocks all external dependencies, including the Ranker.
    """
    # Configure mocks to return mock objects
    mock_embeddings.return_value = MagicMock()
    mock_chroma.return_value.as_retriever.return_value = MagicMock()
    mock_llm.return_value = MagicMock()
    mock_ranker.return_value = MagicMock()  # Mock the Ranker instance

    # Create the RAG chain
    rag_chain = create_rag_chain()

    # Assert that the created object is of the correct type
    assert isinstance(rag_chain, RunnableSequence)

    # Assert that our external dependencies were called (initialized)
    mock_chroma.assert_called_once()
    mock_embeddings.assert_called_once()
    mock_llm.assert_called_once()
    mock_ranker.assert_called_once()
