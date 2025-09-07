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
fetch_arxiv_metadata.py: Fetches initial paper metadata from the arXiv API.

This module serves as the primary entry point for the data ingestion pipeline.
It is responsible for querying the arXiv API based on a search query defined in
the project settings and retrieving the foundational metadata for a corpus of
research papers.

The script's main output is `data/processed/arxiv_metadata.json`. This file acts
as the "source of truth" or "blueprint" for all subsequent data processing
steps, including PDF downloading, concept extraction, and enriched metadata
fetching from other APIs. It contains a list of paper objects, each with its
title, authors, and a clean, unique arXiv ID.
"""

# src/data_processing/fetch_arxiv_metadata

import json
import os
from typing import Any

import arxiv
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

# logger = setup_logging(__name__, "metadata_fetcher")
logger = setup_logging(__name__)


def fetch_and_save_metadata(query: str, num_papers: int, output_path: str) -> None:
    """
    Searches arXiv for papers, extracts structured metadata and
    saves it to a JSON file.
    Args:
        query (str): The search query for arXiv.
        num_papers (int): The maximum number of papers to fetch.
        output_path (str): The file path to save the JSON output.
    """

    logger.info(f"Starting metadata fetch for query: '{query}'")

    try:
        search = arxiv.Search(query=query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
    except Exception as e:
        logger.error(f" An error occurred while fetchhng results from arXiv: {e}")
        return

    if not results:
        logger.warning("No papers found for the given query.")
        return
    logger.info(f"Found {len(results)} papers. Extracting metadata...")

    papers_metadata: list[dict[str, Any]] = []
    for paper in tqdm(results, desc="Fetching Metadata"):
        # The arxiv_id is the lst part of the entry_id URL
        arxiv_id = paper.entry_id.split("/")[-1]

        papers_metadata.append(
            {
                "title": paper.title,
                "arxiv_id": arxiv_id,
                "authors": [author.name for author in paper.authors],
            }
        )
    # Ensure the outpur directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the metadata to a JSON file.
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved metadata for {len(papers_metadata)} papers to {output_path}")
    except OSError as e:
        logger.error(f"Failed to write metadata to file {output_path}: {e}")


if __name__ == "__main__":
    print("Fetching arxiv metadata.")

    METADATA_OUTPUT_PATH = os.path.join(settings.PROCESSED_DATA_PATH, "arxiv_metadata.json")

    fetch_and_save_metadata(
        query=settings.SEARCH_QUERY,
        num_papers=settings.MAX_RESULTS,
        output_path=METADATA_OUTPUT_PATH,
    )
