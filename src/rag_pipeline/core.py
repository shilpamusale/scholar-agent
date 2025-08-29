# src/rag_pipeline/core.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "rag_pipeline")

def create_rag_chain(vector_store_path: Optional[str] = None):
    """
    Creates the core RAG chain.

    Args:
        vector_store_path (Optional[str]): Path to the vector store. 
                                           If None, uses the path from settings.
    """
    logger.info("Creating the RAG chain...")
    
    # Use the provided path or fall back to the settings file
    db_path = vector_store_path if vector_store_path else settings.VECTOR_STORE_PATH
    logger.info(f"Loading vector store from: {db_path}")

    embedding_model = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma(
        persist_directory=db_path, 
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever()
    logger.info("Retriever created successfully.")

    template = """
    You are an expert AI research assistant. Your task is to answer questions based on the context provided.
    Use only the information from the context to answer the question. If the answer is not in the context, say "I don't know".
    Be concise and clear in your response.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL_NAME, temperature=0)
    logger.info(f"LLM initialized with model: {settings.LLM_MODEL_NAME}")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain created successfully.")
    
    return rag_chain
