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
knowledge_graph_tool.py: The implementation of the KnowledgeGraphTool.

This tool bridges the gap between the agent's natural language processing
capabilities and the structured knowledge stored in the Neo4j database.
"""

import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase

import configs.settings as settings
from configs.text_to_cypher import CYPHER_GENERATION_PROMPT, GRAPH_SCHEMA
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "knowledge_graph_tool")


class KnowledgeGraphTool:
    """
    A tool for an agent to query a Neo4j knowledge graph.
    """

    def __init__(self):
        """Initializes connections to the LLM and the Neo4j database."""
        logger.info("Initializing KnowledgeGraphTool...")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME_ADVANCED,
            temperature=0,
        )

        self.prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_PROMPT,
        )

        try:
            URI = os.getenv("NEO4J_URI", "")
            AUTH = (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
            self.driver = GraphDatabase.driver(URI, auth=AUTH)

            logger.info("Neo4j connection established.")

        except Exception as e:
            logger.error(f" Failed to connect to knowledge graph: {e}")
            self.driver = None

    def execute(self, question: str) -> str:
        """
        The main entry point for the tool. An agent calls this method.
        It takes a natural language question and returns a formatted string result.
        """
        if not self.driver:
            return "Error: Database connection is not available."

        logger.info(f" Executing tool with question: '{question}'")
        cypher_query = self._generate_cypher(question)

        if "Error" in cypher_query:
            logger.warning("LLM could not generate Cypher query.")
            return "Error: The question could not be translated into a database query."

        logger.info(f" Generated Cypher query: {cypher_query}")

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query).data()
                logger.info(f"Query returned {len(result)} records.")
                return self.format_result(result)
        except Exception as e:
            logger.error(f"An error occurred during Cypher query execution: {e}")
            return f"Error executing database query: {e}"

    def _generate_cypher(self, question: str) -> str:
        """
        A helper method to create cypher query from the question.
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"schema": GRAPH_SCHEMA, "question": question})
        return response.content.strip()

    def format_result(self, result: str) -> str:
        """
        Formats the list of dictionary results from Neo4j into a string.
        """
        if not result:
            return "No results found."
        formatted_lines = []
        for record in result:
            line_parts = []
            for key, value in record.items():
                line_parts.append(f"{key}: {value}")
            formatted_lines.append(", ".join(line_parts))
        return "\n".join(formatted_lines)
