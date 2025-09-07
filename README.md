# ScholarAgent: An Advanced Multi-Agent Research Assistant

[![CI/CD Status](https://github.com/shilpamusale/scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/shilpamusale/scholar-agent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shilpamusale/scholar-agent/graph/badge.svg)](https://codecov.io/gh/shilpamusale/scholar-agent)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

ScholarAgent is a sophisticated multi-agent system designed to perform deep, relational reasoning over a corpus of scientific papers. It moves beyond standard RAG by building and querying a dynamic Knowledge Graph, allowing it to answer complex questions that require synthesizing information across multiple documents and their relationships.

---

## Demo

![ScholarAgent Demo MP4](assets/scholar_agent_demo.mp4)  
*(A demonstration of the agent answering a complex query using the Knowledge Graph tool.)*

---

## Key Features

* **Automated Knowledge Graph Construction:** A `Makefile`-driven pipeline that automatically fetches papers from arXiv, enriches them with data from Semantic Scholar, extracts key concepts, and builds a comprehensive Neo4j Knowledge Graph.
* **Intelligent Multi-Agent System:** Built with LangGraph, the system uses a Manager agent to intelligently route complex queries to specialized tools.
* **Hybrid Toolset for Deep Reasoning:**
    * **Advanced RAG Tool:** For content-based questions, using a retrieve-then-rerank pipeline for high-quality context.
    * **Knowledge Graph Tool:** For relational questions, using a powerful `gemini-1.5-pro` model to translate natural language into precise Cypher database queries.
* **Tiered LLM Strategy:** Utilizes the efficient `gemini-1.5-flash` for general tasks and the powerful `gemini-1.5-pro` for high-stakes reasoning, balancing performance and cost.
* **Fully Tested and Type-Hinted:** A robust test suite built with `pytest` and a modern, type-hinted codebase enforced by `pre-commit` hooks.

---

## Architecture Overview

The system is split into two core components: an offline Data Pipeline that builds the knowledge base, and an online Agentic System that uses it to answer questions.

```mermaid
graph TD
    subgraph "Offline: Data Pipeline"
        direction LR
        A[External Sources <br> arXiv & Semantic Scholar] --> B{Data Processing Scripts};
        B --> C[Neo4j Knowledge Graph];
        B --> D[ChromaDB Vector Store];
    end

    subgraph "Online: Agentic System"
        direction LR
        E[User Query] --> F{Manager Agent};
        F -- routes to --> G[Knowledge Graph Tool];
        F -- routes to --> H[Advanced RAG Tool];
        G -- queries --> C;
        H -- queries --> D;
        I[Generator Agent]
        G --> I;
        H --> I;
        I --> J[Final Answer];
    end

    classDef source fill:#FFD580,stroke:#666,stroke-width:1.5px,color:#222;
    classDef process fill:#A8E6A3,stroke:#666,stroke-width:1.5px,color:#222;
    classDef storage fill:#9EC9FF,stroke:#666,stroke-width:1.5px,color:#222;
    classDef output fill:#D7B3FF,stroke:#666,stroke-width:1.5px,color:#222;

    class A,E source;
    class B,F,G,H,I process;
    class C,D storage;
    class J output;

```

---

## Tech Stack

<table>
  <tr>
    <td align="center" width="96">
      <a href="https://www.python.org/">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" height="48" alt="Python" />
      </a>
      <br>Python
    </td>
    <td align="center" width="96">
      <a href="https://cloud.google.com/vertex-ai/docs/generative-ai/model-garden/gemini-sdk-overview">
        <img src="https://avatars.githubusercontent.com/u/1342004?s=200&v=4" width="48" height="48" alt="Google Gemini" />
      </a>
      <br>Google Gemini
    </td>
    <td align="center" width="96">
      <a href="https://www.langchain.com/">
        <img src="https://avatars.githubusercontent.com/u/103254248?s=200&v=4" width="48" height="48" alt="LangChain" />
      </a>
      <br>LangChain
    </td>
    <td align="center" width="96">
      <a href="https://neo4j.com/">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/neo4j/neo4j-original.svg" width="48" height="48" alt="Neo4j" />
      </a>
      <br>Neo4j
    </td>
     <td align="center" width="96">
      <a href="https://www.trychroma.com/">
        <img src="https://avatars.githubusercontent.com/u/126588431?s=200&v=4" width="48" height="48" alt="ChromaDB" />
      </a>
      <br>ChromaDB
    </td>
     <td align="center" width="96">
      <a href="https://docs.pytest.org/">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytest/pytest-original.svg" width="48" height="48" alt="Pytest" />
      </a>
      <br>Pytest
    </td>
  </tr>
</table>

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

**Prerequisites:**

* Python 3.12+
* Poetry (for dependency management)
* A running Neo4j instance (e.g., via Docker)
* A Google AI API Key

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your_github_username>/scholar-agent.git
    cd scholar-agent
    ```

2.  **Create a `.env` file:**
    Copy the example environment file and add your credentials.
    ```bash
    cp .env.example .env
    # Now, edit the .env file with your API keys and database URI
    ```

3.  **Install dependencies:**
    ```bash
    poetry install
    ```

4.  **Build the Knowledge Graph:**
    This command will run the entire data pipeline. This may take some time.
    ```bash
    make all
    ```

---

## Usage

Once the knowledge graph is built, you can ask the agent questions from the command line.

**Example 1: Relational Query (Knowledge Graph Tool)**
```bash
python main.py "Who are the most cited authors on the topic of 'sparse autoencoders'?"
```
**Example 2: Content Query (RAG Tool)**
```bash
python main.py "Summarize the abstract of the paper The Interpretable Dictionary in Sparse Coding'"
```
**Example 3: Complex Hybrid Query**
```bash
python main.py "How are researchers at Anthropic using dictionary learning for interpretability, particularly in relation to sparse autoencoders?"
```

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.