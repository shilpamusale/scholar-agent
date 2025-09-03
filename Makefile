# Makefile for the Scholar-Agent project

.PHONY: all data s2-metadata concepts populate-graph clean clean-data clean-raw

# Default target: running 'make' will run the full pipeline
all: populate-graph

# ====================================================================================
# DATA PIPELINE COMMANDS
# ====================================================================================

# This target ensures all base data sources are fetched
data: data/processed/arxiv_metadata.json data/raw

# Rule to create the arXiv metadata file.
data/processed/arxiv_metadata.json: src/data_processing/fetch_arxiv_metadata.py
	@echo "--> Fetching arXiv metadata..."
	python -m src.data_processing.fetch_arxiv_metadata

# Rule to download the raw PDFs.
data/raw: src/data_processing/downloader.py
	@echo "--> Downloading raw PDF files..."
	python -m src.data_processing.downloader

# NEW: Rule to create the Semantic Scholar metadata file.
# This is the crucial missing step.
s2-metadata: data/processed/s2_metadata.json

data/processed/s2_metadata.json: src/data_processing/fetch_s2_metadata.py data/processed/arxiv_metadata.json
	@echo "--> Fetching enriched Semantic Scholar metadata..."
	python -m src.data_processing.fetch_s2_metadata

# Rule to extract concepts from downloaded PDFs
concepts: data/processed/paper_concepts.json

data/processed/paper_concepts.json: src/data_processing/extract_concepts.py data/raw
	@echo "--> Extracting concepts from PDFs..."
	python -m src.data_processing.extract_concepts

# UPDATED: Rule to populate the graph from all processed data
populate-graph: data/processed/s2_metadata.json data/processed/paper_concepts.json
	@echo "--> Populating the knowledge graph from all metadata..."
	python -m src.data_processing.populate_graph

# ====================================================================================
# UTILITY COMMANDS
# ====================================================================================

clean:
	@echo "--> Cleaning all generated data and caches..."
	rm -f data/processed/*.json
	rm -f data/raw/*.pdf
	rm -f logs/*.log
	find . -type d -name "__pycache__" -exec rm -r {} +

clean-data:
	@echo "--> Cleaning up processed data files and logs..."
	rm -f data/processed/*.json
	rm -f logs/*.log
	find . -type d -name "__pycache__" -exec rm -r {} +

clean-raw:
	@echo "--> Cleaning up raw downloaded PDF files..."
	rm -f data/raw/*.pdf
```
