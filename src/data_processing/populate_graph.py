# src/data_processing/populate_graph.py

import json
import os
import re
from typing import Any, Dict

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

    def create_paper(self, s2_id: str, title: str, arxiv_id: str, year: int) -> None:
        query = """
        MERGE (p:Paper {s2_id: $s2_id})
        ON CREATE SET p.title = $title, p.arxiv_id = $arxiv_id, p.year = $year
        ON MATCH SET p.title = $title, p.arxiv_id = $arxiv_id, p.year = $year
        """
        # Using self.driver.execute_query for simplicity and consistency
        self.driver.execute_query(query, s2_id=s2_id, title=title, arxiv_id=arxiv_id, year=year)

    def create_author(self, name: str) -> None:
        self.driver.execute_query("MERGE (a:Author {name: $name})", name=name)

    def link_author_to_paper(self, author_name: str, paper_s2_id: str) -> None:
        query = """
        MATCH (a:Author {name: $author_name})
        MATCH (p:Paper {s2_id: $paper_s2_id})
        MERGE (a)-[:AUTHORED_BY]->(p)
        """
        self.driver.execute_query(query, author_name=author_name, paper_s2_id=paper_s2_id)

    def link_paper_to_citation(self, source_s2_id: str, cited_s2_id: str) -> None:
        query = """
        MATCH (p1:Paper {s2_id: $source_s2_id})
        MATCH (p2:Paper {s2_id: $cited_s2_id})
        MERGE (p1)-[:CITES]->(p2)
        """
        self.driver.execute_query(query, source_s2_id=source_s2_id, cited_s2_id=cited_s2_id)

    def create_concept(self, concept_name: str) -> None:
        self.driver.execute_query("MERGE (c:Concept {name: $name})", name=concept_name)

    def link_paper_to_concept(self, paper_s2_id: str, concept_name: str) -> None:
        query = """
        MATCH (p:Paper {s2_id: $paper_s2_id})
        MATCH (c:Concept {name: $concept_name})
        MERGE (p)-[:DISCUSSES]->(c)
        """
        self.driver.execute_query(query, paper_s2_id=paper_s2_id, concept_name=concept_name)


def load_json_data(file_path: str) -> Any:
    """Loads and returns data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}. Please run 'make' to generate it.")
        return None

def run_population_pipeline() -> None:
    logger.info("Starting final graph population pipeline...")
    load_dotenv()
    URI = os.getenv("NEO4J_URI", "")
    AUTH = (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
    
    s2_papers_map = load_json_data(settings.PROCESSED_DATA_PATH / "s2_metadata.json") or {}
    concepts_map = load_json_data(settings.PROCESSED_DATA_PATH / "paper_concepts.json") or {}

    logger.info(f"Loaded S2 metadata for {len(s2_papers_map)} papers.")
    logger.info(f"Loaded concepts for {len(concepts_map)} papers.")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        populator = GraphPopulator(driver)
        
        all_cited_s2_ids = set()
        # CORRECTED: Iterate over the dictionary's values
        for paper_data in s2_papers_map.values():
            for ref in paper_data.get('references', []):
                if ref and ref.get('paperId'):
                    all_cited_s2_ids.add(ref['paperId'])
        
        logger.info(f"Creating {len(all_cited_s2_ids)} placeholder nodes for cited papers...")
        for s2_id in tqdm(all_cited_s2_ids, desc="Creating Cited Placeholders"):
            populator.create_paper(s2_id, "Title Unknown", "N/A", 0)

        logger.info("Populating full graph data...")
        # CORRECTED: Iterate over the dictionary's items (key, value)
        for arxiv_id, paper_data in tqdm(s2_papers_map.items(), desc="Populating Full Graph"):
            s2_id = paper_data.get('paperId')
            
            populator.create_paper(
                s2_id=s2_id,
                title=paper_data.get('title'),
                arxiv_id=arxiv_id,
                year=paper_data.get('year')
            )
            
            for author in paper_data.get('authors', []):
                populator.create_author(author['name'])
                populator.link_author_to_paper(author['name'], s2_id)
            
            for ref in paper_data.get('references', []):
                populator.link_paper_to_citation(s2_id, ref['paperId'])
            
            if arxiv_id in concepts_map:
                for concept in concepts_map[arxiv_id]:
                    populator.create_concept(concept)
                    populator.link_paper_to_concept(s2_id, concept)

    logger.info("Full graph population pipeline finished successfully.")

if __name__ == "__main__":
    run_population_pipeline()