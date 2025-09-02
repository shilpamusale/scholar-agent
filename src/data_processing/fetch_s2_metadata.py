# src/data_processing/fetch_s2_metadata.py

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from configs import settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "s2_metadata_fetcher")


def clean_arxiv_id(arxiv_id: str) -> str:
    """Removes the version number from an arXiv ID
    (e.g., '1706.03762v2' -> '1706.03762').
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def fetch_paper_details(arxiv_id: str) -> dict[str, Any] | None:
    """
    Fetches detailed paper metadata, including citations,
    from the Semantic Scholar API.

    Args:
        arxiv_id: The clean, unversioned arXiv
                    ID of the paper to look up.

    Returns:
        A dictionary containing the paper's metadata,
            or None if an error occurs.
    """
    url = settings.S2_API_URL.format(arxiv_id=arxiv_id)
    params = {"fields": settings.S2_API_FIELDS}
    try:
        response = requests.get(url, params=params, timeout=15)  # Increased timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error for arXiv:{arxiv_id}: {e}")
        if e.response.status_code == 404:
            logger.warning(f"Paper arXiv:{arxiv_id} not found on Semantic Scholar.")
        # If we are rate-limited, wait for a longer period before continuing
        if e.response.status_code == 429:
            logger.warning("Rate limit hit. Waiting for 60 seconds...")
            time.sleep(60)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for arXiv:{arxiv_id}: {e}")
        return None


def fetch_all_metadata(input_path: Path, output_path: Path):
    """
    Loads a list of papers, fetches their metadata from Semantic Scholar,
    and saves the enriched data.
    """
    logger.info(f"Loading initial metadata from {input_path}...")
    try:
        with open(input_path) as f:
            initial_papers = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not read or parse input file {input_path}: {e}")
        return

    enriched_papers: list[dict[str, Any]] = []
    logger.info(
        f"Fetching enriched metadata for {len(initial_papers)} "
        f"papers from Semantic Scholar..."
    )

    for paper in tqdm(initial_papers, desc="Fetching S2 Data"):
        original_arxiv_id = paper.get("arxiv_id")
        if not original_arxiv_id:
            continue

        # --- FIX: Clean the arXiv ID before using it ---
        cleaned_id = clean_arxiv_id(original_arxiv_id)

        details = fetch_paper_details(cleaned_id)
        if details:
            # Add the original arXiv ID back in for consistency if needed elsewhere
            details["originalArxivId"] = original_arxiv_id
            enriched_papers.append(details)

        # Respect the API rate limit. 3.1 seconds is a safer margin.
        time.sleep(3.1)

    logger.info(f"Successfully fetched metadata for {len(enriched_papers)} papers.")
    logger.info(f"Saving enriched metadata to {output_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(enriched_papers, f, indent=2)
        logger.info("Save complete.")
    except OSError as e:
        logger.error(f"Could not write to output file {output_path}: {e}")


if __name__ == "__main__":
    fetch_all_metadata(
        input_path=settings.PROCESSED_DATA_PATH / "arxiv_metadata.json",
        output_path=settings.PROCESSED_DATA_PATH / "s2_metadata.json",
    )
