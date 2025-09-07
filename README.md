# ScholarAgent 🔬

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

ScholarAgent is an advanced, multi-agent research assistant designed to reason over a corpus of scientific papers. It leverages a sophisticated RAG pipeline, a knowledge graph, and a multi-agent architecture to answer complex questions and uncover novel insights from technical documents.

---

## 🚀 Demo

![ScholarAgent Demo GIF](assets/scholar_agent_demo.gif)
*(A demonstration of the agent answering a complex query using the Knowledge Graph tool.)*

---

## ✨ Key Features

* **Advanced RAG Pipeline:** Implements a two-stage retrieval process that combines a fast, high-recall vector search (ChromaDB) with a powerful, high-precision cross-encoder model for re-ranking. This ensures the most relevant context is provided to the language model.

* **Multi-Agent Architecture:** Utilizes a collaborative team of AI agents built with LangGraph. The system features a **Manager** for task decomposition, a specialized **Search Agent** for information retrieval, and a **Writer Agent** for synthesizing coherent answers.

* **Knowledge Graph Integration:** Automatically builds and queries a Neo4j knowledge graph from research papers. This allows the agent to answer complex questions about relationships between papers, authors, and concepts that are impossible with vector search alone.

* **Polished & Transparent CLI:** An interactive command-line interface built with `rich` that visualizes the agent's entire thought process, including tool selections, generated Cypher queries, database results in tables, and the final synthesized answer.

---

## 🏗️ System Architecture

![System Architecture Diagram](assets/architecture.png)
*(Diagram showing the flow from user query to the Manager agent, which delegates tasks to the RAG and Knowledge Graph tools before synthesizing a final answer.)*

---

## 🛠️ Tech Stack

* **LLMs & Agents:** LangChain, LangGraph, Google Gemini
* **Data & ML:** PyTorch, `sentence-transformers`
* **Databases:** Neo4j (Graph), ChromaDB (Vector)
* **Core:** Python 3.12+
* **Tooling:** `rich` (for CLI), `pre-commit`, `ruff`, Poetry

---

## ⚙️ Setup and Installation

Follow these steps to get ScholarAgent running on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/scholar-agent.git](https://github.com/your-username/scholar-agent.git)
cd scholar-agent
```

### 2. Configure Environment Variables
Copy the example environment file and fill in your API keys and database credentials.
```bash
cp .env.example .env
```
You will need to add your credentials for:
* `GOOGLE_API_KEY`
* `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

### 3. Install Dependencies
This project uses Poetry for dependency management.
```bash
poetry install
```

### 4. Ingest Data
Before running the agent, you must populate the knowledge bases. Place your research papers (PDFs) in the `data/raw` directory and run the ingestion script.
```bash
python scripts/ingest.py
```
This script will process the PDFs, build the ChromaDB vector store, and populate the Neo4j knowledge graph.

---

## 💻 Usage

Run the agent from the command line by providing a query in quotes.

### Example 1: RAG Query
```bash
python main.py "Summarize the key findings of the paper 'Attention Is All You Need'"
```

### Example 2: Knowledge Graph Query
```bash
python main.py "Who are the most cited authors on the topic of 'sparse autoencoders'?"
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
