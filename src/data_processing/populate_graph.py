# src/data_processing/populate_graph.py

import json
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "graph_populator")


class GraphPopulator:
    """Populates the Neo4j database with paper, author, and concept data."""

    def __init__(self, driver):
        self.driver = driver

    def create_paper(self, paper_data: dict[str, Any]) -> None:
        """Creates or merges a Paper node."""
        query = """
        MERGE (p:Paper {s2_id: $s2_id})
        ON CREATE SET p.title = $title, p.year = $year, p.arxiv_id = $arxiv_id
        """
        with self.driver.session() as session:
            session.run(
                query,
                s2_id=paper_data.get("paperId"),
                title=paper_data.get("title"),
                year=paper_data.get("year"),
                arxiv_id=paper_data.get("arxivId"),
            )

    def create_author(self, author_data: dict[str, Any]) -> None:
        """Creates or merges an Author node."""
        query = "MERGE (a:Author {name: $name})"
        with self.driver.session() as session:
            session.run(query, name=author_data.get("name"))

    def link_author_to_paper(self, author_name: str, paper_id: str) -> None:
        """Creates an AUTHORED_BY relationship."""
        query = """
        MATCH (a:Author {name: $author_name})
        MATCH (p:Paper {s2_id: $paper_id})
        MERGE (a)-[:AUTHORED_BY]->(p)
        """
        with self.driver.session() as session:
            session.run(query, author_name=author_name, paper_id=paper_id)

    def create_concept(self, concept_name: str) -> None:
        """Creates or merges a Concept node."""
        query = "MERGE (c:Concept {name: $name})"
        with self.driver.session() as session:
            session.run(query, name=concept_name)

    def link_paper_to_concept(self, paper_id: str, concept_name: str) -> None:
        """Creates a DISCUSSES relationship."""
        query = """
        MATCH (p:Paper {s2_id: $paper_id})
        MATCH (c:Concept {name: $concept_name})
        MERGE (p)-[:DISCUSSES]->(c)
        """
        with self.driver.session() as session:
            session.run(query, paper_id=paper_id, concept_name=concept_name)

    def close(self):
        self.driver.close()


def run_graph_population_pipeline():
    """Main function to run the entire graph population pipeline."""
    logger.info("Starting graph population pipeline...")

    # Load the processed data
    with open(settings.PROCESSED_DATA_PATH / "s2_metadata.json") as f:
        papers_metadata = json.load(f)
    with open(settings.PROCESSED_DATA_PATH / "paper_concepts.json") as f:
        paper_concepts = json.load(f)

    logger.info(f"Loaded metadata for {len(papers_metadata)} papers.")
    logger.info(f"Loaded concepts for {len(paper_concepts)} papers.")

    load_dotenv()
    URI = os.getenv("NEO4J_URI", "")
    AUTH = (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
    if not URI:
        raise ValueError("NEO4J_URI environment variable not set.")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        populator = GraphPopulator(driver)

        for paper_data in tqdm(papers_metadata, desc="Populating Graph"):
            s2_id = paper_data.get("paperId")
            if not s2_id:
                continue

            # Create Paper node
            populator.create_paper(paper_data)

            # Create Author nodes and link them
            for author in paper_data.get("authors", []):
                populator.create_author(author)
                populator.link_author_to_paper(author.get("name"), s2_id)

            # Create Concept nodes and link them
            arxiv_id = paper_data.get("arxivId")
            if arxiv_id in paper_concepts:
                for concept in paper_concepts[arxiv_id]:
                    populator.create_concept(concept)
                    populator.link_paper_to_concept(s2_id, concept)

    logger.info("Graph population pipeline finished successfully.")


if __name__ == "__main__":
    run_graph_population_pipeline()
