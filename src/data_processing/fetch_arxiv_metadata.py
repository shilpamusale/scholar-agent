# src/data_processing/fetch_arxiv_metadata

import json
import os
from typing import Any

import arxiv
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "metadata_fetcher")


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
        search = arxiv.Search(
            query=query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance
        )
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
        logger.info(
            f"Successfully saved metadata for {len(papers_metadata)}"
            f" papers to {output_path}"
        )
    except OSError as e:
        logger.error(f"Failed to write metadata to file {output_path}: {e}")


if __name__ == "__main__":
    print("Fetching arxiv metadata.")

    METADATA_OUTPUT_PATH = os.path.join(
        settings.PROCESSED_DATA_PATH, "arxiv_metadata.json"
    )

    fetch_and_save_metadata(
        query=settings.SEARCH_QUERY,
        num_papers=settings.MAX_RESULTS,
        output_path=METADATA_OUTPUT_PATH,
    )
