# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of a greed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
populate_graph.py: Populates the Neo4j database from processed data files.

This module is the final and most critical step in the data ingestion pipeline.
It acts as the master assembler, taking all the processed and enriched data
sources and building the complete, interconnected knowledge graph in Neo4j.

The script performs the following steps:
1.  Loads the enriched paper/citation data from `s2_metadata.json`.
2.  Loads the extracted concept data from `paper_concepts.json`.
3.  Intelligently "joins" these disparate data sources using their common
    arXiv ID as a key.
4.  Populates the Neo4j database by creating all nodes (`Paper`, `Author`,
    `Concept`) and all relationships (`AUTHORED_BY`, `CITES`, `DISCUSSES`).

The result is a fully populated, queryable knowledge graph that serves as a
primary tool for the Scholar-Agent.
"""

# src/data_processing/populate_graph.py

import json
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

# logger = setup_logging(__name__, "populate_graph")
logger = setup_logging(__name__)


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
        with open(file_path) as f:
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
    len_s2 = len(s2_papers_map)
    len_cn = len(concepts_map)
    logger.info(f"Loaded S2 metadata for {len_s2} papers.")
    logger.info(f"Loaded concepts for {len_cn} papers.")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        populator = GraphPopulator(driver)

        all_cited_s2_ids = set()
        # CORRECTED: Iterate over the dictionary's values
        for paper_data in s2_papers_map.values():
            for ref in paper_data.get("references", []):
                if ref and ref.get("paperId"):
                    all_cited_s2_ids.add(ref["paperId"])

        logger.info(f"Creating {len(all_cited_s2_ids)} placeholder nodes for cited papers...")
        for s2_id in tqdm(all_cited_s2_ids, desc="Creating Cited Placeholders"):
            populator.create_paper(s2_id, "Title Unknown", "N/A", 0)

        logger.info("Populating full graph data...")
        # CORRECTED: Iterate over the dictionary's items (key, value)
        for arxiv_id, paper_data in tqdm(s2_papers_map.items(), desc="Populating Full Graph"):
            s2_id = paper_data.get("paperId")

            populator.create_paper(
                s2_id=s2_id,
                title=paper_data.get("title"),
                arxiv_id=arxiv_id,
                year=paper_data.get("year"),
            )

            for author in paper_data.get("authors", []):
                populator.create_author(author["name"])
                populator.link_author_to_paper(author["name"], s2_id)

            for ref in paper_data.get("references", []):
                populator.link_paper_to_citation(s2_id, ref["paperId"])

            if arxiv_id in concepts_map:
                for concept in concepts_map[arxiv_id]:
                    populator.create_concept(concept)
                    populator.link_paper_to_concept(s2_id, concept)

    logger.info("Full graph population pipeline finished successfully.")


if __name__ == "__main__":
    run_population_pipeline()
