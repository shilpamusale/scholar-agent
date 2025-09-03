# src/data_processing/fetch_s2_metadata.py

import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "fetch_s2_metadata")


def _clean_arxiv_id(arxiv_id: str) -> str:
    """Removes the version number from an arXiv ID
    (e.g., '1706.03762v1' -> '1706.03762')."""
    return arxiv_id.split("v")[0]


def fetch_single_paper_metadata(
    session: requests.Session, arxiv_id: str
) -> dict[str, Any] | None:
    """Fetches enriched metadata for a single
    paper from the Semantic Scholar API."""
    clean_id = _clean_arxiv_id(arxiv_id)
    url = f"{settings.S2_API_URL}/paper/arXiv:{clean_id}"
    try:
        response = session.get(url, params={"fields": settings.S2_API_FIELDS})
        # Raises an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Paper arXiv:{clean_id} not found on Semantic Scholar.")
        else:
            logger.error(f"HTTP Error for arXiv:{clean_id}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"A network error occurred for arXiv:{clean_id}: {e}")
    return None


def fetch_all_metadata() -> None:
    """
    Main function to fetch metadata for
    all papers in the corpus,
    respecting API rate limits and using an API key.
    """
    load_dotenv()
    api_key: str | None = os.getenv("S2_API_KEY")
    if not api_key:
        logger.error("S2_API_KEY not found in environment variables. Aborting.")
        return

    headers: dict[str, str] = {"x-api-key": api_key}

    input_path = settings.PROCESSED_DATA_PATH / "arxiv_metadata.json"
    output_path = settings.PROCESSED_DATA_PATH / "s2_metadata.json"

    logger.info(f"Loading initial metadata from {input_path}...")
    try:
        with open(input_path) as f:
            arxiv_data: list[dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"Input file not found: {input_path}. Please run 'make data' first."
        )
        return

    enriched_papers: list[dict[str, Any]] = []
    logger.info(
        f"Fetching enriched metadata for {len(arxiv_data)} "
        f"papers from Semantic Scholar..."
    )

    with requests.Session() as session:
        session.headers.update(headers)
        for item in tqdm(arxiv_data, desc="Fetching S2 Data"):
            arxiv_id = item.get("arxiv_id")
            if not arxiv_id:
                continue

            paper_metadata = fetch_single_paper_metadata(session, arxiv_id)
            if paper_metadata:
                enriched_papers.append(paper_metadata)

            # Respect the API rate limit
            time.sleep(1.1)

    with open(output_path, "w") as f:
        json.dump(enriched_papers, f, indent=2)
    logger.info(
        f"Successfully fetched and saved metadata for {len(enriched_papers)}"
        f" papers to {output_path}"
    )


if __name__ == "__main__":
    fetch_all_metadata()
