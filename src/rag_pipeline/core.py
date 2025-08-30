from typing import Dict

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


def create_rag_chain(vector_store_path: str = settings.VECTOR_STORE_PATH):
    """
    Creates and returns a RAG (Retrieval-Augmented Generation) chain.
    The chain is designed to take a question, retrieve relevant documents,
    re-rank them for precision, and then generate an answer using an LLM.

    Args:
        vector_store_path (str): The path to the persisted ChromaDB vector store.

    Returns:
        A runnable LangChain object representing the RAG chain.
    """
    logger.info("Creating the RAG chain with re-ranking...")

    # --- 1. Embedding Model ---
    embedding_model = SentenceTransformerEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

    # --- 2. Vector Store and Base Retriever ---
    vector_store = Chroma(
        persist_directory=vector_store_path, embedding_function=embedding_model
    )
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_TOP_K}
    )
    logger.info("Base retriever created successfully.")

    # --- 3. Cross-Encoder Re-ranker ---
    ranker = Ranker(model_name=settings.CROSS_ENCODER_MODEL_NAME)
    logger.info("Flashrank re-ranker initialized.")

    # --- 4. Custom Re-ranking and Formatting Function ---
    def rerank_and_format_context(data: Dict) -> str:
        """
        Takes the output of the base retriever, re-ranks it, and formats
        the top documents into a single string for the LLM context.
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
        return "\n\n---\n\n".join(top_docs_content)

    # --- 5. The Prompt Template ---
    template = """
    You are an expert research assistant.
    Your goal is to provide a clear and concise answer to the user's question,
    based ONLY on the following context.
    Do not add any information that is not present in the context.
    If the answer is not in the context, say "Sorry, I couldn’t find that
    information in the provided context."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created.")

    # --- 6. The LLM ---
    llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL_NAME)
    logger.info(f"LLM initialized with model: {settings.LLM_MODEL_NAME}")

    # --- 7. The Final RAG Chain for Evaluation ---
    # This chain passes the retrieved context through to the final output
    # so that RAGAS can evaluate it.
    retriever_chain = RunnableParallel(
        {"context": base_retriever, "question": RunnablePassthrough()}
    )

    # The processing chain that takes the retrieved
    # context and question to generate an answer
    answer_chain = (
        RunnableLambda(rerank_and_format_context) | prompt | llm | StrOutputParser()
    )

    # The final chain that combines
    # everything and returns a dictionary
    rag_chain = retriever_chain | RunnableParallel(
        {"answer": answer_chain, "context": lambda x: x["context"]}
    )

    logger.info("RAG chain for evaluation created successfully.")
    return rag_chain
