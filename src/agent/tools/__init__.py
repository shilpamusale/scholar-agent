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
__init__.py: Turns the 'tools' directory into a Python package.

This file also acts as the central
factory for creating and providing tools
to the AI agent. By placing the
`get_tools` function here, we can easily
import it from other parts of the
application using `from src.agent.tools
import get_tools`.
"""

from langchain_core.tools import Tool

from src.agent.schemas import ToolInputSchema
from src.rag_pipeline.core import create_rag_chain
from src.utils.logging_config import setup_logging

# Import the new tool class from its module within this package
from .knowledge_graph_tool import KnowledgeGraphTool

logger = setup_logging(__name__, "agent_tools")


def get_tools() -> list:
    """
    Initializes and returns a list of tools available to the agent.
    This function is called once when the agent graph is being compiled.
    """
    logger.info("Initializing tools...")

    # 1. The RAG Tool
    rag_chain = create_rag_chain()
    research_paper_search_tool = Tool(
        name="research_paper_search",
        func=rag_chain.invoke,
        description=(
            "Searches and retrieves information "
            "from the text of research papers. "
            "Use this for questions about the content, "
            "abstract, or findings of a specific paper."
        ),
        args_schema=ToolInputSchema,
    )
    logger.info("Research paper search tool created.")

    # 2. The Knowledge Graph Tool
    knowledge_graph_tool_instance = KnowledgeGraphTool()
    knowledge_graph_query_tool = Tool(
        name="knowledge_graph_query",
        func=knowledge_graph_tool_instance.execute,
        description=(
            "Queries the knowledge graph to find "
            "relationships between papers, authors, and concepts. "
            "Use this for questions about citations, collaborations, "
            "influential authors, or connections between topics."
        ),
        args_schema=ToolInputSchema,
    )
    logger.info("Knowledge graph query tool created.")

    return [research_paper_search_tool, knowledge_graph_query_tool]
