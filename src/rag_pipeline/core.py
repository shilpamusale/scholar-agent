# src/rag_pipeline/core.py
import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import(
    SentenceTransformerEmbeddings
)
from langchain_google_genai import(
    ChatGoogleGenerativeAI
)
from langchain.prompts import(
    ChatPromptTemplate
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import configurations and logging utilities
import configs.settings as settings
from src.utils.logging_config import(
    setup_logging
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup Logger
logger = setup_logging(__name__, "rag_pipeline")

def create_rag_chain():
    """
    Creates the core Retrieval-Augmented Generation (RAG)
    chain using Gemini
    """
    logger.info("Creating the RAG chain...")

    # 1. Load vector store and create retriever
    embedding_model = SentenceTransformerEmbeddings(
        model_name = settings.EMBEDDING_MODEL_NAME
    )
    vector_store = Chroma(
        persist_directory = settings.VECTOR_STORE_PATH,
        embedding_function = embedding_model
    )

    retriever = vector_store.as_retriever()
    logger.info("Retriever created successfully.")

    # 2. Define the prompt template

    template = """
    You are an expert AI research assistant. 
    Your task is to answer questions based on the context provided.
    Use only the information from the context to answer the question. 
    If the answer is not in the context, say 
    "Sorry, I couldn’t find that information in the provided context.".
    Be concise and clear in your response. 

    Context: {context}
    Question: {question}

    Answer: 
    """

    prompt = ChatPromptTemplate.from_template(template)
    logger.info("Prompt template created.")

    # 3. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model = settings.LLM_MODEL_NAME,
        temperature = 0
        ) 
    logger.info(f"LLM initialized with model: {settings.LLM_MODEL_NAME}")

    # Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever, 
         "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
    )

    logger.info("RAG chain created successfully.")

    return rag_chain


