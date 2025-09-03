# Makefile for the Scholar-Agent project

.PHONY: all populate-graph clean clean-data clean-raw

# Default target: running 'make' will run the full pipeline
all: populate-graph

# ====================================================================================
# DATA PIPELINE COMMANDS
# ====================================================================================

# Rule to create the arXiv metadata file.
data/processed/arxiv_metadata.json: src/data_processing/fetch_arxiv_metadata.py
	@echo "--> Fetching arXiv metadata..."
	@python -m src.data_processing.fetch_arxiv_metadata

# Rule to download all PDFs. This now depends on the metadata file existing first.
# Using a "stamp" file to represent the completion of the directory download.
data/raw/.download_stamp: src/data_processing/downloader.py data/processed/arxiv_metadata.json
	@echo "--> Downloading raw PDF files..."
	@python -m src.data_processing.downloader
	@touch data/raw/.download_stamp

# Rule to create the Semantic Scholar metadata file.
data/processed/s2_metadata.json: src/data_processing/fetch_s2_metadata.py data/processed/arxiv_metadata.json
	@echo "--> Fetching enriched Semantic Scholar metadata..."
	@python -m src.data_processing.fetch_s2_metadata

# Rule to extract concepts from downloaded PDFs. This depends on the download stamp.
data/processed/paper_concepts.json: src/data_processing/extract_concepts.py data/raw/.download_stamp
	@echo "--> Extracting concepts from PDFs..."
	@python -m src.data_processing.extract_concepts

# Final rule to populate the graph. This now depends on all final data files.
populate-graph: data/processed/s2_metadata.json data/processed/paper_concepts.json
	@echo "--> Populating the knowledge graph from all metadata..."
	@python -m src.data_processing.populate_graph

# ====================================================================================
# UTILITY COMMANDS
# ====================================================================================

clean:
	@echo "--> Cleaning all generated data and caches..."
	@rm -f data/processed/*.json
	@rm -f data/raw/*
	@rm -f logs/*.log
	@find . -type d -name "__pycache__" -exec rm -rf {} +
