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
tool_result.py: Normalises tool output and decides whether a route succeeded.

Each tool in this system returns a different shape: the knowledge-graph tool
returns a JSON string, the RAG chain returns a dict of ``{answer, context}``.
Before the graph can decide whether a route produced usable grounding, those
shapes have to be reduced to a single verdict.

This module does that, and it does it in pure Python -- no LangChain, no LLM,
no network. That is deliberate. The routing policy of an agent is the part most
likely to break silently in production, so it is kept as a pure function that
can be unit-tested exhaustively without mocking a model.

Two concepts:

* ``ToolResult`` -- a normalised envelope carrying a status (``ok``, ``empty``,
  ``error``), the payload, and a human-readable detail string that is fed back
  to the manager when a route is retried.
* ``decide_after_tool`` -- the routing policy. Given a status, a hop count, and
  the set of routes already burned, it returns the next edge to take.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

# --- Route names returned by the routing policy --------------------------------

RETRY = "retry"
GENERATE_ANSWER = "generate_answer"


class ToolStatus(str, Enum):
    """The three outcomes a tool invocation can have.

    ``EMPTY`` is the important one: the tool ran correctly and returned nothing
    useful. That is not an exception, so it will not surface as one, but it is
    exactly the case where the agent should try the other route rather than ask
    the generator to synthesise an answer out of nothing.
    """

    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


# The RAG prompt instructs the model to emit this sentence when the retrieved
# passages do not contain the answer. The template uses a typographic
# apostrophe (U+2019), so text is normalised before matching.
_NO_CONTEXT_MARKERS = (
    "sorry, i couldn't find that information",
    "could not be translated into a database query",
)


def _normalise(text: str) -> str:
    """Lowercase and fold typographic apostrophes for robust marker matching."""
    return text.replace("\u2019", "'").strip().lower()


@dataclass
class ToolResult:
    """A normalised tool outcome, ready to be serialised into a ToolMessage."""

    tool: str
    status: ToolStatus
    payload: Any = None
    detail: str = ""

    @property
    def usable(self) -> bool:
        """True when the tool produced grounding the generator can rely on."""
        return self.status is ToolStatus.OK

    def to_json(self) -> str:
        """Serialise to the envelope the graph puts on the message log."""
        return json.dumps(
            {
                "tool": self.tool,
                "status": self.status.value,
                "detail": self.detail,
                "payload": self.payload,
            },
            indent=2,
            default=str,
        )


def _document_summary(doc: Any) -> dict[str, Any]:
    """Reduce a LangChain Document to a JSON-serialisable summary."""
    content = getattr(doc, "page_content", str(doc))
    metadata = getattr(doc, "metadata", {}) or {}
    return {
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page"),
        "excerpt": content[:300],
    }


def classify_knowledge_graph_output(raw: Any, tool: str = "knowledge_graph_query") -> ToolResult:
    """Classify the JSON string returned by ``KnowledgeGraphTool.execute``.

    Three failure modes are distinguished, because they call for different
    retry messages: the tool could not reach Neo4j or the query blew up
    (``ERROR``), the LLM could not produce Cypher for the question (``ERROR``),
    or valid Cypher ran and matched nothing (``EMPTY``).
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ToolResult(tool, ToolStatus.ERROR, {"raw": raw}, "Tool returned non-JSON output.")
    elif isinstance(raw, dict):
        data = raw
    else:
        return ToolResult(tool, ToolStatus.ERROR, {"raw": str(raw)}, "Unexpected tool return type.")

    if "error" in data:
        return ToolResult(tool, ToolStatus.ERROR, data, str(data["error"]))

    if not data.get("database_result"):
        return ToolResult(
            tool,
            ToolStatus.EMPTY,
            data,
            "The Cypher query executed successfully but matched no records in the graph.",
        )

    return ToolResult(tool, ToolStatus.OK, data)


def classify_rag_output(raw: Any, tool: str = "research_paper_search") -> ToolResult:
    """Classify the ``{answer, context}`` dict returned by the RAG chain.

    An answer with no retrieved passages behind it, or one carrying the
    prompt's explicit no-context sentence, is treated as ``EMPTY`` -- the chain
    worked, the corpus just did not hold the answer.
    """
    if isinstance(raw, dict):
        answer = str(raw.get("answer", ""))
        documents = raw.get("context") or []
    else:
        answer, documents = str(raw), []

    payload = {"answer": answer, "sources": [_document_summary(d) for d in documents]}

    if not documents:
        return ToolResult(tool, ToolStatus.EMPTY, payload, "The retriever returned no passages for this query.")

    if any(marker in _normalise(answer) for marker in _NO_CONTEXT_MARKERS):
        return ToolResult(
            tool,
            ToolStatus.EMPTY,
            payload,
            "The retrieved passages did not contain an answer to this question.",
        )

    if not answer.strip():
        return ToolResult(tool, ToolStatus.EMPTY, payload, "The tool produced an empty answer.")

    return ToolResult(tool, ToolStatus.OK, payload)


_CLASSIFIERS = {
    "knowledge_graph_query": classify_knowledge_graph_output,
    "research_paper_search": classify_rag_output,
}


def classify_tool_output(tool_name: str, raw: Any) -> ToolResult:
    """Dispatch to the classifier registered for ``tool_name``.

    An unregistered tool is an ``ERROR`` rather than a crash: the graph should
    degrade to the other route, not take down the request.
    """
    classifier = _CLASSIFIERS.get(tool_name)
    if classifier is None:
        return ToolResult(
            tool_name,
            ToolStatus.ERROR,
            {"raw": str(raw)},
            f"No result classifier is registered for tool '{tool_name}'.",
        )
    return classifier(raw)


def decide_after_tool(
    status: ToolStatus | str,
    hops: int,
    failed_tools: list[str],
    total_tools: int,
    max_hops: int,
) -> str:
    """The routing policy applied after every tool execution.

    Returns ``RETRY`` to send control back to the manager for a different
    route, or ``GENERATE_ANSWER`` to proceed to synthesis.

    The agent falls back only when all three of these hold:

    1. the route did not produce usable grounding,
    2. the hop budget has not been spent, and
    3. at least one route has not yet been tried.

    Conditions 2 and 3 are what stop the cycle. Without them a corpus that
    simply does not contain the answer would bounce the agent between routes
    until LangGraph's recursion limit fired.
    """
    status = ToolStatus(status)

    if status is ToolStatus.OK:
        return GENERATE_ANSWER
    if hops >= max_hops:
        return GENERATE_ANSWER
    if len(set(failed_tools)) >= total_tools:
        return GENERATE_ANSWER
    return RETRY
