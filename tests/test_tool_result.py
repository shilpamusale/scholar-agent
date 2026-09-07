"""
test_tool_result.py: Tests for tool-output classification and the fallback
routing policy.

These are the tests that guard the fallback behaviour. They need no API key,
no Neo4j instance, and no model call, because `tool_result.py` holds the policy
as pure functions -- which is the reason it is a separate module.
"""

import json

import pytest

from src.agent.tool_result import (
    GENERATE_ANSWER,
    RETRY,
    ToolStatus,
    classify_knowledge_graph_output,
    classify_rag_output,
    classify_tool_output,
    decide_after_tool,
)


class FakeDocument:
    """Stands in for a LangChain Document without importing LangChain."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# --- Knowledge graph classification -------------------------------------------


def test_kg_result_with_records_is_ok():
    raw = json.dumps({"cypher_query": "MATCH (a:Author) RETURN a.name", "database_result": [{"a.name": "Olah"}]})
    result = classify_knowledge_graph_output(raw)
    assert result.status is ToolStatus.OK
    assert result.usable


def test_kg_query_matching_nothing_is_empty_not_ok():
    """A valid query that matches no nodes raises nothing. It must still be
    classified as a failed route, or the generator receives an empty result set
    and improvises around it."""
    raw = json.dumps({"cypher_query": "MATCH (p:Paper {title: 'Nonexistent'}) RETURN p", "database_result": []})
    result = classify_knowledge_graph_output(raw)
    assert result.status is ToolStatus.EMPTY
    assert not result.usable
    assert "matched no records" in result.detail


def test_kg_error_payload_is_error():
    raw = json.dumps({"error": "Database connection is not available."})
    result = classify_knowledge_graph_output(raw)
    assert result.status is ToolStatus.ERROR
    assert "connection" in result.detail


def test_kg_untranslatable_question_is_error():
    raw = '{"error": "The question could not be translated into a database query."}'
    assert classify_knowledge_graph_output(raw).status is ToolStatus.ERROR


def test_kg_non_json_output_is_error_not_crash():
    result = classify_knowledge_graph_output("Error")
    assert result.status is ToolStatus.ERROR
    assert result.payload["raw"] == "Error"


# --- RAG classification --------------------------------------------------------


def test_rag_result_with_passages_is_ok():
    raw = {
        "answer": "Polysemantic neurons activate on unrelated inputs.",
        "context": [FakeDocument("...", {"source": "paper.pdf", "page": 3})],
    }
    result = classify_rag_output(raw)
    assert result.status is ToolStatus.OK
    assert result.payload["sources"][0]["source"] == "paper.pdf"


def test_rag_no_context_sentence_is_empty():
    """The RAG prompt emits this sentence with a typographic apostrophe. The
    classifier must match it regardless of apostrophe form."""
    raw = {
        "answer": "Sorry, I couldn\u2019t find that information in the provided context.",
        "context": [FakeDocument("unrelated passage")],
    }
    result = classify_rag_output(raw)
    assert result.status is ToolStatus.EMPTY


def test_rag_with_no_retrieved_documents_is_empty():
    result = classify_rag_output({"answer": "Something confident.", "context": []})
    assert result.status is ToolStatus.EMPTY
    assert "no passages" in result.detail


def test_rag_blank_answer_is_empty():
    result = classify_rag_output({"answer": "   ", "context": [FakeDocument("passage")]})
    assert result.status is ToolStatus.EMPTY


# --- Dispatch ------------------------------------------------------------------


def test_unregistered_tool_is_error_not_crash():
    result = classify_tool_output("some_hallucinated_tool", "whatever")
    assert result.status is ToolStatus.ERROR


def test_envelope_serialises_to_json():
    raw = {"answer": "grounded", "context": [FakeDocument("passage")]}
    envelope = json.loads(classify_rag_output(raw).to_json())
    assert envelope["status"] == "ok"
    assert envelope["tool"] == "research_paper_search"
    assert envelope["payload"]["answer"] == "grounded"


# --- Routing policy ------------------------------------------------------------


def test_successful_route_goes_straight_to_generator():
    assert decide_after_tool(ToolStatus.OK, hops=1, failed_tools=[], total_tools=2, max_hops=2) == GENERATE_ANSWER


@pytest.mark.parametrize("status", [ToolStatus.EMPTY, ToolStatus.ERROR])
def test_failed_first_route_falls_back(status):
    decision = decide_after_tool(
        status,
        hops=1,
        failed_tools=["knowledge_graph_query"],
        total_tools=2,
        max_hops=2,
    )
    assert decision == RETRY


def test_hop_budget_stops_the_cycle():
    """Even with a route left untried, the budget terminates the loop."""
    decision = decide_after_tool(
        ToolStatus.EMPTY,
        hops=2,
        failed_tools=["knowledge_graph_query"],
        total_tools=3,
        max_hops=2,
    )
    assert decision == GENERATE_ANSWER


def test_exhausting_every_route_stops_the_cycle():
    decision = decide_after_tool(
        ToolStatus.ERROR,
        hops=2,
        failed_tools=["knowledge_graph_query", "research_paper_search"],
        total_tools=2,
        max_hops=5,
    )
    assert decision == GENERATE_ANSWER


def test_repeated_failures_of_one_route_do_not_count_as_exhaustion():
    """Deduplication matters: two failures of the same tool leave the other
    route untried, so the agent should still fall back."""
    decision = decide_after_tool(
        ToolStatus.ERROR,
        hops=1,
        failed_tools=["knowledge_graph_query", "knowledge_graph_query"],
        total_tools=2,
        max_hops=3,
    )
    assert decision == RETRY


def test_policy_accepts_status_as_string_from_state():
    """State stores the status as a plain string for serialisability."""
    assert decide_after_tool("ok", hops=1, failed_tools=[], total_tools=2, max_hops=2) == GENERATE_ANSWER
