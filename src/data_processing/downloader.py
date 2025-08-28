# src/data_processing/downloader.py

import arxiv
import os
from tqdm import tqdm

# Import our custom logging and settings
from src.utils.logging_config import setup_logging
import configs.settings as settings

# Get the logger instance for this specific module
logger = setup_logging(__name__, "downloader")

def download_papers(query: str, num_papers: int, output_dir: str):
    """
    Searches for papers on arXiv based on a query and downloads them as PDFs.

    Args:
        query (str): The search query for arXiv.
        num_papers (int): The maximum number of papers to download.
        output_dir (str): The directory to save the downloaded PDFs.
    """
    logger.info(f"Starting download for query: '{query}'")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory '{output_dir}' is ready.")
    except OSError as e:
        logger.error(f"Error creating directory {output_dir}: {e}")
        return

    search = arxiv.Search(
        query=query,
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    try:
        results = list(search.results())
    except Exception as e:
        logger.error(f"An error occurred while fetching results from arXiv: {e}")
        return
    
    if not results:
        logger.warning("No papers found for the given query.")
        return

    logger.info(f"Found {len(results)} papers. Starting download...")

    for paper in tqdm(results, desc="Downloading Papers"):
        try:
            safe_title = "".join(c for c in paper.title if c.isalnum() or c in (' ', '.', '_')).rstrip()
            filename = f"{safe_title}.pdf"
            
            paper.download_pdf(dirpath=output_dir, filename=filename)
            logger.info(f"Successfully downloaded '{filename}'")

        except Exception as e:
            logger.error(f"Failed to download paper: {paper.title}. Error: {e}")

    logger.info("Download process finished.")


if __name__ == "__main__":
    download_papers(
        query=settings.SEARCH_QUERY, 
        num_papers=settings.NUM_PAPERS_TO_DOWNLOAD,
        output_dir=settings.RAW_DATA_PATH
    )
