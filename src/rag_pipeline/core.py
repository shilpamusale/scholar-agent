# src/rag_pipeline/core.py

from typing import Dict

from flashrank import Ranker, RerankRequest
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "rag_pipeline")


def create_rag_chain() -> Runnable:
    """
    Creates the complete RAG chain with a re-ranking step.

    Returns:
        Runnable: An executable LangChain RAG chain.
    """
    logger.info("Creating the RAG chain with re-ranking...")

    # --- 1. Initial Retriever ---
    embedding_model = SentenceTransformerEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME
    )
    vector_store = Chroma(
        persist_directory=str(settings.VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    logger.info("Base retriever created successfully.")

    # --- 2. Re-ranker ---
    ranker = Ranker(
        model_name=settings.CROSS_ENCODER_MODEL_NAME,
        cache_dir="/workspaces/scholar-agent/.flashrank_cache",
    )

    def rerank_and_format_context(data: Dict) -> Dict:
        """Reranks docs and formats them into a single context string."""
        query = data["question"]
        docs = data["context"]  # This is a list of Document objects

        passages_for_reranking = [
            {"id": i, "text": doc.page_content} for i, doc in enumerate(docs)
        ]

        rerank_request = RerankRequest(query=query, passages=passages_for_reranking)

        reranked_passages = ranker.rerank(rerank_request)

        top_docs_content = [result["text"] for result in reranked_passages[:4]]

        return {
            "context": "\n\n".join(top_docs_content),
            "question": query,
        }

    # --- 3. Prompt Template ---
    template = """You are an AI research assistant. Use the following
    pieces of retrieved context to answer the question at the end.
    If you don't know the answer, just say that you don't know.
    Be concise and informative.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created.")

    # --- 4. LLM ---
    llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL_NAME)
    logger.info(f"LLM initialized with model: {settings.LLM_MODEL_NAME}")

    # --- 5. The Full Chain with Re-ranking ---
    rag_chain = (
        {"context": base_retriever, "question": RunnablePassthrough()}
        | RunnableLambda(rerank_and_format_context)
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain with re-ranking created successfully.")
    return rag_chain
