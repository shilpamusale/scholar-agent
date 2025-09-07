# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,

"""
downloader.py: A standalone script for downloading research papers from arXiv.

This module serves as a critical first step in the data ingestion pipeline for the
Scholar-Agent project. It queries the arXiv API for a specified search term
(defined in `configs.settings`) and downloads the resulting papers as PDF files.

The primary output of this script is a collection of PDF documents stored in the
`data/raw` directory. Each PDF is robustly named using its unique, version-stripped
arXiv ID (e.g., '2405.08366.pdf') to ensure a stable and unambiguous link
between the raw file and its metadata for all downstream processing, such as
concept extraction.

This script is designed to be executed as part of the automated `Makefile`
pipeline but can also be run directly for development purposes.
"""
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

    search = arxiv.Search(query=query, max_results=num_papers, sort_by=arxiv.SortCriterion.Relevance)
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
