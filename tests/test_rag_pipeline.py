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
tests/test_rag_pipeline.py: Unit tests for the RAG chain creation.
"""

from unittest.mock import MagicMock, patch

from langchain_core.runnables import Runnable

from src.rag_pipeline.core import create_rag_chain  # noqa: E402


@patch("configs.settings.get_google_api_key")
@patch("src.rag_pipeline.core.Ranker")
@patch("src.rag_pipeline.core.ChatGoogleGenerativeAI")
@patch("src.rag_pipeline.core.SentenceTransformerEmbeddings")
@patch("src.rag_pipeline.core.Chroma")
def test_create_rag_chain(
    mock_chroma,
    mock_embeddings,
    mock_llm,
    mock_ranker,
    mock_get_api_key,
):
    """
    Tests that the RAG chain is created successfully and is a runnable object.
    This test mocks all external dependencies to ensure it runs in isolation.
    """
    # Configure mocks to return mock objects
    mock_embeddings.return_value = MagicMock()
    mock_chroma.return_value.as_retriever.return_value = MagicMock()
    mock_llm.return_value = MagicMock()
    mock_ranker.return_value = MagicMock()
    mock_get_api_key.return_value = "fake-api-key"

    # Create the RAG chain
    rag_chain = create_rag_chain()

    # Assert that the created object is a LangChain runnable
    assert isinstance(rag_chain, Runnable), "The RAG chain should be a runnable object."

    # Assert that our mocks were called, confirming the chain was assembled
    mock_embeddings.assert_called_once()
    mock_chroma.assert_called_once()
    mock_llm.assert_called_once()
    mock_ranker.assert_called_once()
    mock_get_api_key.assert_called_once()
