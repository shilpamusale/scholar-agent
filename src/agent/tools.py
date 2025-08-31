# src/agent/tools.py

from langchain_core.tools import Tool

from src.rag_pipeline.core import create_rag_chain
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "agent_tools")


def get_tools() -> list:
    """
    Initializes and returns a list of tools available to the agent.
    This function is called once when the agent graph is being compiled.
    """
    logger.info("Initializing tools...")

    # Create an instance of our RAG chain to be used as a tool
    rag_chain = create_rag_chain()

    # Define the research paper search tool
    research_paper_search_tool = Tool(
        name="research_paper_search",
        func=rag_chain.invoke,
        description=(
            "Searches and retrieves information from a collection of research papers "
            "on AI, machine learning, and interpretability."
        ),
    )
    logger.info("Research paper search tool created.")

    return [research_paper_search_tool]
