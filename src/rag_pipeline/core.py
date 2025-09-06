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
core.py: Factory for creating the advanced RAG-with-reranking chain.

This module contains the primary logic for constructing the Retrieval-Augmented
Generation (RAG) pipeline for the Scholar-Agent. This is not a standard RAG
chain; it implements an advanced retrieve-then-rerank pattern to improve the
quality and relevance of the context provided to the Large Language Model.

The pipeline is built using the LangChain Expression Language (LCEL) and
consists of the following key stages:
1.  **Initial Retrieval:** A high-recall, semantic search is performed against
    a ChromaDB vector store to fetch a large set of candidate documents.
2.  **Re-ranking:** The initial set of documents is passed to a more
    sophisticated Cross-Encoder model (via FlashRank) which re-ranks the
    documents for precision and relevance to the specific query.
3.  **Prompting and Generation:** The top N re-ranked documents are formatted
    and injected into a prompt, which is then passed to a Google Gemini model
    to synthesize the final, grounded answer.

The `create_rag_chain` function serves as a factory, encapsulating this
complexity and returning a runnable LangChain object that can be used as a
tool by the agentic system.
"""

# src/rag_pipeline/core.py
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "rag_pipeline")


def create_rag_chain(
    vector_store_path: str = str(settings.VECTOR_STORE_PATH),
):
    """
    Creates and returns a RAG (Retrieval-Augmented Generation) chain.
    """
    logger.info("Creating the RAG chain with re-ranking...")

    embedding_model = SentenceTransformerEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

    vector_store = Chroma(
        persist_directory=vector_store_path, embedding_function=embedding_model
    )
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_TOP_K}
    )
    logger.info("Base retriever created successfully.")

    ranker = Ranker(
        model_name=settings.CROSS_ENCODER_MODEL_NAME, cache_dir="/tmp/flashrank_cache"
    )
    logger.info("Flashrank re-ranker initialized.")

    def rerank_and_prepare_for_prompt(data: dict) -> dict:
        """
        Takes retriever output, re-ranks it, and formats for the prompt.
        """
        query = data["question"]
        docs = data["context"]
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_passages = ranker.rerank(rerank_request)
        top_docs_content = [
            result["text"] for result in reranked_passages[: settings.RERANKER_TOP_N]
        ]
        logger.info(f"Re-ranked {len(docs)} documents down to {len(top_docs_content)}.")
        return {
            "context": "\n\n---\n\n".join(top_docs_content),
            "question": query,
        }

    template = """
    You are an expert research assistant.
    Your goal is to provide a clear and
    concise answer to the user's question,
    based ONLY on the following context.
    Do not add any information that is not
    present in the context.
    If the answer is not in the context,
    say "Sorry, I couldn’t find that information in the provided context."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created.")

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME,
        temperature=0,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        google_api_key=settings.GOOGLE_API_KEY,
    )
    logger.info(f"LLM initialized with model: {settings.LLM_MODEL_NAME}")

    retriever_chain = RunnableParallel(
        {"context": base_retriever, "question": RunnablePassthrough()}
    )

    answer_chain = (
        RunnableLambda(rerank_and_prepare_for_prompt) | prompt | llm | StrOutputParser()
    )

    rag_chain = retriever_chain | RunnableParallel(
        {"answer": answer_chain, "context": lambda x: x["context"]}
    )

    logger.info("RAG chain for evaluation created successfully.")
    return rag_chain
