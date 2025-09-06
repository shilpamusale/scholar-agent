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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
prompts.py: A central repository for all system prompts used by the agentic system.

Externalizing prompts from the core agent logic allows for easier experimentation,
tuning, and management of the agent's behavior and personality.
"""

# This prompt empowers the manager to choose the best tool.
# The LLM will automatically see the names
# and descriptions of the tools
# provided in `tools.py` and use them to make its decision.
MANAGER_PROMPT = (
    "You are an expert research assistant and project manager. "
    "Your goal is to answer the user's query by selecting the "
    "most appropriate tool for the job. You have two tools available: "
    "`research_paper_search` for finding specific "
    "information within documents, and "
    "`knowledge_graph_query` for understanding "
    "the relationships between papers, authors, and concepts. "
    "Carefully analyze the user's question "
    "and call the tool that is best "
    "suited to answer it. Do not try to "
    "answer the question from your own knowledge."
)

GENERATOR_PROMPT = (
    "You are an expert research assistant. "
    "Your task is to synthesize a clear and "
    "concise answer to the user's question based "
    "*only* on the provided context from the tool. "
    "Structure your answer logically. "
    "If the context contains multiple points, "
    "synthesize them into a coherent response. "
    "Do not add any information or "
    "opinions that are not explicitly stated in the context."
)
