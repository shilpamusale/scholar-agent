# .PHONY defines targets that aren't actual files.
.PHONY: all data clean-data populate-graph

# Default command to run when you just type "make"
all: data populate-graph

# ====================================================================================
# DATA PIPELINE COMMANDS
# ====================================================================================

# Orchestrator target to run the full data preparation pipeline.
data: data/processed/arxiv_metadata.json data/raw

# Rule to create the metadata file, run as a module.
data/processed/arxiv_metadata.json: src/data_processing/fetch_arxiv_metadata.py configs/settings.py
	@echo "--> Fetching arXiv metadata..."
	@mkdir -p data/processed
	python -m src.data_processing.fetch_arxiv_metadata

# Rule to download the raw PDFs, run as a module.
data/raw: src/data_processing/downloader.py configs/settings.py
	@echo "--> Downloading raw PDF files..."
	@mkdir -p data/raw
	python -m src.data_processing.downloader

# ====================================================================================
# GRAPH DATABASE COMMANDS
# ====================================================================================

# Target to populate the Neo4j database, run as a module.
populate-graph: data/processed/arxiv_metadata.json
	@echo "--> Populating the knowledge graph from metadata..."
	python -m src.data_processing.populate_graph

# ====================================================================================
# UTILITY COMMANDS
# ====================================================================================

# Target to clean all generated data for a fresh start.
clean-data:
	@echo "--> Cleaning up generated data directories..."
	@rm -rf data/processed
	@rm -rf data/raw
