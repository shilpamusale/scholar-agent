# src/data_processing/downloader.py

import os
import re

import arxiv
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "downloader")


def clean_arxiv_id(raw_id: str) -> str:
    """Removes the version number from an arXiv ID."""
    match = re.search(r"(\d{4}\.\d{4,5})", raw_id)
    return match.group(1) if match else raw_id


def download_papers(query: str, num_papers: int, output_dir: str):
    """
    Searches arXiv and downloads papers, naming them by their arXiv ID.
    """
    logger.info(f"Starting download for query: '{query}'")
    os.makedirs(output_dir, exist_ok=True)

    search = arxiv.Search(
        query=query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance
    )
    results = list(search.results())

    if not results:
        logger.warning("No papers found for the given query.")
        return

    logger.info(f"Found {len(results)} papers. Starting download...")
    for paper in tqdm(results, desc="Downloading Papers"):
        try:
            # Use the clean arXiv ID as the filename
            arxiv_id = clean_arxiv_id(paper.entry_id)
            filename = f"{arxiv_id}.pdf"

            paper.download_pdf(dirpath=output_dir, filename=filename)
            logger.info(f"Successfully downloaded '{filename}'")
        except Exception as e:
            logger.error(f"Failed to download paper: {paper.title}. Error: {e}")
    logger.info("Download process finished.")


if __name__ == "__main__":
    download_papers(
        query=settings.SEARCH_QUERY,
        num_papers=settings.MAX_RESULTS,
        output_dir=settings.RAW_DATA_PATH,
    )
