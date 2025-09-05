# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
tools.py: A factory for creating and providing tools to the AI agent.

This module is responsible for initializing all the capabilities (tools) that
the agentic system can use to perform tasks and answer questions. It acts as a
bridge between the complex, underlying data pipelines (like the RAG chain) and
the agent's decision-making framework.

Each tool is wrapped in a standardized `langchain_core.tools.Tool` object,
which provides a name and a natural language description. The agent's LLM uses
these descriptions to intelligently select the appropriate tool for a given
task.

This script will be expanded to include additional tools, such as the
GraphQueryTool for interacting with the Neo4j knowledge graph.
"""

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
