# src/data_processing/populate_graph
import json
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j.graph import Node
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging


def load_metadata_from_file(file_path: str) -> list[dict[str, Any]]:
    """
    Loads paper metadata from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents a paper.
    """
    print(f"--> Loading metadata from {file_path}...")
    try:
        with open(file_path) as f:
            data = json.load(f)
        print(f"--> Successfully loaded {len(data)} paper records.")
        return data
    except FileNotFoundError:
        print(f"--> ERROR: Metadata file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"--> ERROR: Could not decode JSON from {file_path}")
        return []


class GraphPopulator:
    """
    Populates the Neo4j database with paper and author data.
    """

    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    def create_paper(self, title: str, arxiv_id: str) -> Node:
        """Creates or merges a Paper node in the graph."""
        query = (
            "MERGE (p:Paper {arxiv_id: $arxiv_id}) "
            "ON CREATE SET p.title = $title "
            "RETURN p"
        )
        with self.driver.session() as session:
            result = session.run(query, arxiv_id=arxiv_id, title=title)
            record = result.single()
            if record:
                return record[0]
            raise Exception("Could not create or find paper node.")

    def create_author(self, name: str) -> Node:
        """Creates or merges an Author node in the graph."""
        query = "MERGE (a:Author {name: $name}) RETURN a"
        with self.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            if record:
                return record[0]
            raise Exception("Could not create or find author node.")

    def link_author_to_paper(self, author_name: str, paper_arxiv_id: str) -> None:
        """Creates an AUTHORED_BY relationship between an Author and a Paper."""
        query = (
            "MATCH (a:Author {name: $author_name}) "
            "MATCH (p:Paper {arxiv_id: $paper_arxiv_id}) "
            "MERGE (a)-[r:AUTHORED_BY]->(p) "
            "RETURN r"
        )
        with self.driver.session() as session:
            result = session.run(
                query, author_name=author_name, paper_arxiv_id=paper_arxiv_id
            )
            if not result.single():
                raise ValueError(
                    f"Failed to create link. Author '{author_name}' or "
                    f"Paper '{paper_arxiv_id}' may not exist."
                )


if __name__ == "__main__":
    load_dotenv()
    URI: str = os.getenv("NEO4J_URI", "")
    AUTH = (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
    logger = setup_logging(__name__, "populate_graph")
    if not URI:
        raise ValueError("NEO4J_URI environment variable not set.")

    # --- Load Real Data from File ---
    METADATA_PATH = os.path.join(settings.PROCESSED_DATA_PATH, "arxiv_metadata.json")
    papers_data = load_metadata_from_file(METADATA_PATH)

    if not papers_data:
        logger.warning("No data loaded. Exiting population script.")
    else:
        try:
            with GraphDatabase.driver(URI, auth=AUTH) as driver:
                driver.verify_connectivity()
                logger.info("Connection verified.")
                populator = GraphPopulator(driver)

                logger.info(f"Populating graph with {len(papers_data)} papers...")
                for paper_data in tqdm(papers_data, desc="Populating Graph"):
                    populator.create_paper(paper_data["title"], paper_data["arxiv_id"])
                    for author_name in paper_data["authors"]:
                        populator.create_author(author_name)
                        populator.link_author_to_paper(
                            author_name, paper_data["arxiv_id"]
                        )

                logger.info("\nGraph population complete.")

        except Exception as e:
            logger.error(f"An error occurred during graph population: {e}")
