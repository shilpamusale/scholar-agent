# src/data_processing/populate_graph.py

import json
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "populate_graph")


class GraphPopulator:
    """Populates the Neo4j database with all processed data."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    def create_paper(self, s2_id: str, title: str, arxiv_id: str | None) -> None:
        query = """
        MERGE (p:Paper {s2_id: $s2_id})
        ON CREATE SET p.title = $title, p.arxiv_id = $arxiv_id
        """
        with self.driver.session() as session:
            session.run(query, s2_id=s2_id, title=title, arxiv_id=arxiv_id)

    def create_author(self, name: str) -> None:
        query = "MERGE (a:Author {name: $name})"
        with self.driver.session() as session:
            session.run(query, name=name)

    def link_author_to_paper(self, author_name: str, paper_s2_id: str) -> None:
        query = """
        MATCH (a:Author {name: $author_name})
        MATCH (p:Paper {s2_id: $paper_s2_id})
        MERGE (a)-[:AUTHORED_BY]->(p)
        """
        with self.driver.session() as session:
            session.run(query, author_name=author_name, paper_s2_id=paper_s2_id)

    def link_paper_to_citation(self, source_s2_id: str, cited_s2_id: str) -> None:
        query = """
        MATCH (p1:Paper {s2_id: $source_s2_id})
        MATCH (p2:Paper {s2_id: $cited_s2_id})
        MERGE (p1)-[:CITES]->(p2)
        """
        with self.driver.session() as session:
            session.run(query, source_s2_id=source_s2_id, cited_s2_id=cited_s2_id)

    def create_concept(self, concept_name: str) -> None:
        query = "MERGE (c:Concept {name: $concept_name})"
        with self.driver.session() as session:
            session.run(query, concept_name=concept_name)

    def link_paper_to_concept(self, arxiv_id: str, concept_name: str) -> None:
        query = """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MATCH (c:Concept {name: $concept_name})
        MERGE (p)-[:DISCUSSES]->(c)
        """
        with self.driver.session() as session:
            session.run(query, arxiv_id=arxiv_id, concept_name=concept_name)

    def close(self) -> None:
        self.driver.close()


def load_json_data(file_path: str) -> Any:
    """Loads and returns data from a JSON file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(
            f"Error loading {file_path}: {e}. "
            f"Please run the full 'make' pipeline first."
        )
        return None


def run_population_pipeline() -> None:
    """Main function to run the entire graph population pipeline."""
    logger.info("Starting final graph population pipeline...")
    load_dotenv()
    URI = os.getenv("NEO4J_URI", "")
    AUTH = (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
    if not URI:
        raise ValueError("NEO4J_URI environment variable not set.")

    s2_papers_data = (
        load_json_data(settings.PROCESSED_DATA_PATH / "s2_metadata.json") or []
    )
    concepts_data = (
        load_json_data(settings.PROCESSED_DATA_PATH / "paper_concepts.json") or {}
    )

    # --- THIS IS THE FIX ---
    # The concepts_data is already a dictionary mapping arxiv_id to concepts.
    # We can use it directly.
    concepts_map = concepts_data

    logger.info(f"Loaded S2 metadata for {len(s2_papers_data)} papers.")
    logger.info(f"Loaded concepts for {len(concepts_map)} papers.")

    driver = GraphDatabase.driver(URI, auth=AUTH)
    populator = GraphPopulator(driver)

    try:
        all_cited_s2_ids = set()
        for paper in s2_papers_data:
            for ref in paper.get("references", []):
                if ref and ref.get("paperId"):
                    all_cited_s2_ids.add(ref["paperId"])

        logger.info(
            f"Creating {len(all_cited_s2_ids)} placeholder nodes for cited papers..."
        )
        for s2_id in tqdm(all_cited_s2_ids, desc="Creating Cited Papers"):
            populator.create_paper(s2_id, "Title Unknown", None)

        for paper_data in tqdm(s2_papers_data, desc="Populating Full Graph"):
            arxiv_id = paper_data.get("arxivId")
            s2_id = paper_data.get("paperId")
            title = paper_data.get("title")

            if not s2_id or not title:
                continue

            populator.create_paper(s2_id, title, arxiv_id)
            for author in paper_data.get("authors", []):
                if author and author.get("name"):
                    populator.create_author(author["name"])
                    populator.link_author_to_paper(author["name"], s2_id)

            for ref in paper_data.get("references", []):
                if ref and ref.get("paperId"):
                    populator.link_paper_to_citation(s2_id, ref["paperId"])

            if arxiv_id and arxiv_id in concepts_map:
                # We use .get() to be safe, in case a paper has no concepts
                for concept_name in concepts_map.get(arxiv_id, []):
                    populator.create_concept(concept_name)
                    populator.link_paper_to_concept(arxiv_id, concept_name)

    finally:
        populator.close()
        logger.info("Full graph population pipeline finished successfully.")


if __name__ == "__main__":
    run_population_pipeline()
