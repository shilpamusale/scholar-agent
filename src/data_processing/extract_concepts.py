# src/data_processing/extract_concepts.py

import json
import re
from pathlib import Path

import pke
import spacy
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

import configs.settings as settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "concept_extractor")

POST_FILTER_STOPLIST = {
    "figure",
    "table",
    "model",
    "network",
    "layer",
    "features",
    "section",
    "activations",
    "similar",
    "results",
    "approach",
    "method",
    "system",
    "prediction",
    "correlations",
    "language",
    "disease",
    "speech",
    "introduction",
    "conclusion",
    "references",
    "appendix",
    "acknowledgments",
    "algorithm",
}


class ConceptExtractor:
    """Extracts key concepts from research papers using PKE."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = spacy_model
        if not spacy.util.is_package(self.spacy_model):
            spacy.cli.download(self.spacy_model)
        self.nlp = spacy.load(self.spacy_model)

    def extract_concepts(
        self, document_text: str, top_n: int = 15
    ) -> list[tuple[str, float]]:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(
            input=document_text, language="en", spacy_model=self.nlp
        )
        pos = {"NOUN", "PROPN", "ADJ"}
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.74, method="average")
        raw_keyphrases = extractor.get_n_best(n=100)

        filtered_keyphrases = []
        for phrase, score in raw_keyphrases:
            # Filter for in-text citation patterns
            if re.search(r"\b(et al|p|vol|pp|no|fig)\.?", phrase, re.IGNORECASE):
                continue

            words = phrase.lower().split()
            # Filter for generic boilerplate terms
            if any(word in POST_FILTER_STOPLIST for word in words):
                continue

            # Prioritize multi-word phrases, but allow single-word all-caps acronyms
            if len(words) > 1 or (len(words) == 1 and phrase.isupper()):
                filtered_keyphrases.append((phrase, score))

        return filtered_keyphrases[:top_n]


def run_concept_extraction_pipeline():
    """
    Processes all PDFs, extracts concepts,
    and saves them to a JSON file.
    """
    logger.info("Starting concept extraction pipeline for all downloaded PDFs...")
    pdf_files = list(Path(settings.RAW_DATA_PATH).glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found. Please run the downloader first.")
        return

    extractor = ConceptExtractor()
    all_concepts: dict[str, list[str]] = {}

    for pdf_path in tqdm(pdf_files, desc="Extracting Concepts"):
        try:
            arxiv_id = pdf_path.stem

            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            full_text = " ".join(page.page_content for page in pages)

            concepts_with_scores = extractor.extract_concepts(full_text, top_n=15)
            # We only need the concept text, not the score, for the final JSON
            concepts = [concept for concept, score in concepts_with_scores]

            all_concepts[arxiv_id] = concepts

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}. Error: {e}")

    output_path = settings.PROCESSED_DATA_PATH / "paper_concepts.json"
    logger.info(
        f"Saving extracted concepts for {len(all_concepts)} papers to {output_path}..."
    )
    with open(output_path, "w") as f:
        json.dump(all_concepts, f, indent=2)
    logger.info("Concept extraction pipeline finished successfully.")


if __name__ == "__main__":
    run_concept_extraction_pipeline()
