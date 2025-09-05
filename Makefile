# Makefile for the Scholar-Agent project

# Define .PHONY targets to prevent conflicts with files of the same name.
# These are our explicit, runnable commands.
.PHONY: all clean fetch-arxiv download-pdfs fetch-s2 extract-concepts populate-graph

# Default Goal: Running 'make' will run the 'all' target.
all: populate-graph

# ====================================================================================
# USER-FACING COMMANDS (for step-by-step execution)
# ====================================================================================

fetch-arxiv: data/processed/arxiv_metadata.json
	@echo "~~~~~~~~~~~~~~~~~~~--> ArXiv metadata is up to date."

download-pdfs: data/raw/.download_stamp
	@echo "~~~~~~~~~~~~~~~~~~~--> PDFs are up to date."

fetch-s2: data/processed/s2_metadata.json
	@echo "~~~~~~~~~~~~~~~~~~~--> Semantic Scholar metadata is up to date."

extract-concepts: data/processed/paper_concepts.json
	@echo "~~~~~~~~~~~~~~~~~~~--> Paper concepts are up to date."

# The populate-graph command now depends on its prerequisites.
populate-graph: data/processed/s2_metadata.json data/processed/paper_concepts.json
	@echo "~~~~~~~~~~~~~~~~~~~--> (5/5) Populating the knowledge graph..."
	@python -m src.data_processing.populate_graph

# ====================================================================================
# INTERNAL FILE-BASED DEPENDENCIES (for automation)
# ====================================================================================

# 1. Start with the arXiv metadata.
data/processed/arxiv_metadata.json: src/data_processing/fetch_arxiv_metadata.py
	@echo "--> (1/5) Fetching arXiv metadata..."
	@python -m src.data_processing.fetch_arxiv_metadata

# 2. PDF downloads depend on the arXiv metadata.
data/raw/.download_stamp: src/data_processing/downloader.py data/processed/arxiv_metadata.json
	@echo "--> (2/5) Downloading raw PDF files..."
	@python -m src.data_processing.downloader
	@touch data/raw/.download_stamp

# 3. Concept extraction depends on the PDFs being downloaded.
data/processed/paper_concepts.json: src/data_processing/extract_concepts.py data/raw/.download_stamp
	@echo "--> (3/5) Extracting concepts from PDFs..."
	@python -m src.data_processing.extract_concepts

# 4. Semantic Scholar data also depends on the initial arXiv data.
data/processed/s2_metadata.json: src/data_processing/fetch_s2_metadata.py data/processed/arxiv_metadata.json
	@echo "--> (4/5) Fetching enriched Semantic Scholar metadata..."
	@python -m src.data_processing.fetch_s2_metadata


# ====================================================================================
# UTILITY COMMANDS
# ====================================================================================

clean:
	@echo "--> Cleaning all generated data and caches..."
	@rm -f data/processed/*.json
	@rm -f data/raw/*
	@rm -f logs/*.log
	@find . -type d -name "__pycache__" -exec rm -rf {} +
