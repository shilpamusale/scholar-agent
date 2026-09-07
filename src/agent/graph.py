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
graph.py: Constructs and compiles the core agentic graph using LangGraph.

The graph is a cycle, not a pipeline. A clean pass looks like this:

    manager -> tool_executor -> generator -> END

but when a route fails to produce grounding, `tool_executor` sends control
back to `manager` instead of forward to `generator`:

    manager -> tool_executor -> manager -> tool_executor -> generator -> END

Three mechanisms make that cycle safe:

1.  **Structural exclusion.** A route that has already failed is removed from
    the tool set bound to the manager LLM on the retry pass. The model cannot
    re-select a burned route, because the invalid action is never offered to
    it. This is stronger than instructing the model not to repeat itself.

2.  **A hop budget.** `settings.MAX_TOOL_HOPS` caps total tool executions per
    query, so the cycle terminates even if the exclusion logic is wrong.

3.  **Exhaustion.** Once every registered route has been tried, the graph moves
    to synthesis regardless of remaining budget.

Failure detection lives in `tool_result.py` and is deliberately not exception
based. The most common failure here is a route that succeeds mechanically and
returns nothing useful -- a Cypher query that matches no nodes, a retriever
that surfaces no relevant passages. Neither raises, so a try/except would send
them straight to the generator, which is the bug this graph exists to avoid.
"""

# src/agent/graph.py

import json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

import configs.settings as settings
from configs.prompts import (
    GENERATOR_NO_CONTEXT_PROMPT,
    GENERATOR_PROMPT,
    MANAGER_PROMPT,
    ROUTE_RETRY_DIRECTIVE,
)
from src.agent.tool_result import (
    GENERATE_ANSWER,
    RETRY,
    ToolResult,
    ToolStatus,
    classify_tool_output,
    decide_after_tool,
)
from src.agent.tools import get_tools
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# --- LLM and Tool Initialization (Done Once) ---
tools = get_tools()
TOOL_MAP = {tool.name: tool for tool in tools}

# The manager LLM is held unbound. Tools are bound per invocation in
# `manager_node`, because the available tool set narrows as routes are burned.
base_manager_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    google_api_key=settings.get_google_api_key(),
)
generator_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    google_api_key=settings.get_google_api_key(),
)


# --- Agent Definition ---
class AgentState(TypedDict):
    """State carried through the graph.

    `messages` and `hops` use additive reducers, so a node returns only its
    increment. `failed_tools` accumulates the names of burned routes.
    `last_status` and `last_detail` are last-write-wins and describe the most
    recent tool execution.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    hops: Annotated[int, operator.add]
    failed_tools: Annotated[list[str], operator.add]
    last_status: str
    last_detail: str


def initial_state(query: str) -> AgentState:
    """Builds a fully populated initial state for a user query.

    Every channel is seeded explicitly so nodes can read state without
    defensive lookups on the first pass.
    """
    return {
        "messages": [HumanMessage(content=query)],
        "hops": 0,
        "failed_tools": [],
        "last_status": "",
        "last_detail": "",
    }


def manager_node(state: AgentState) -> dict:
    """Selects a route, binding only the tools that have not yet failed."""
    failed = set(state.get("failed_tools", []))
    available = [tool for tool in tools if tool.name not in failed]

    if not available:
        # Defensive: the routing policy should reach the generator before this
        # can happen. Rebinding the full set keeps the model call valid.
        logger.warning("All routes burned but manager was invoked; rebinding full tool set.")
        available = tools

    prompt = MANAGER_PROMPT
    if failed:
        prompt = (
            MANAGER_PROMPT
            + "\n\n"
            + ROUTE_RETRY_DIRECTIVE.format(
                failed_tool=", ".join(sorted(failed)),
                detail=state.get("last_detail") or "No detail recorded.",
            )
        )
        logger.info(f"Manager retrying. Burned: {sorted(failed)}. Available: {[t.name for t in available]}")

    response = base_manager_llm.bind_tools(available).invoke([HumanMessage(content=prompt)] + state["messages"])
    logger.info(f"Manager response: {response}")
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Executes the selected tool and classifies whether it produced grounding.

    Always returns a normalised envelope on the message log, so downstream
    nodes and the CLI parse one shape regardless of which tool ran.
    """
    logger.info("Tool node executing.")
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    logger.info(f"Executing {tool_name} with args: {tool_args}")

    tool_to_call = TOOL_MAP.get(tool_name)

    if tool_to_call is None:
        result = ToolResult(
            tool=tool_name,
            status=ToolStatus.ERROR,
            detail=f"Tool '{tool_name}' is not registered.",
        )
    else:
        query = tool_args.get("question") or next(iter(tool_args.values()), None)
        if not query:
            result = ToolResult(
                tool=tool_name,
                status=ToolStatus.ERROR,
                detail="The tool was called without a usable question argument.",
            )
        else:
            try:
                raw = tool_to_call.invoke(query)
                result = classify_tool_output(tool_name, raw)
            except Exception as exc:  # noqa: BLE001 - surfaced as a route failure, not a crash
                logger.error(f"Tool '{tool_name}' raised: {exc}", exc_info=True)
                result = ToolResult(
                    tool=tool_name,
                    status=ToolStatus.ERROR,
                    detail=f"The tool raised an exception: {exc}",
                )

    logger.info(f"Tool '{tool_name}' finished with status '{result.status.value}'. {result.detail}")

    return {
        "messages": [ToolMessage(content=result.to_json(), tool_call_id=tool_call["id"])],
        "hops": 1,
        "failed_tools": [] if result.usable else [tool_name],
        "last_status": result.status.value,
        "last_detail": result.detail,
    }


def _original_question(state: AgentState) -> str:
    """The user's question: the first human turn on the message log."""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _latest_tool_output(state: AgentState) -> str:
    """The most recent tool envelope, as a JSON string."""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            return message.text
    return ""


def generator_node(state: AgentState) -> dict:
    """Synthesises the final answer.

    Uses a different prompt when the agent arrives with no grounding, so it
    reports the failure instead of improvising from an error envelope.

    The question and tool output are passed as plain text rather than by
    replaying the message log. The log contains an AIMessage carrying
    tool_calls paired with a ToolMessage, and this model is invoked with no
    tools bound -- providers may return an empty completion when a
    conversation references tool declarations that are absent from the
    request. Flattening to text removes that coupling entirely, and the
    generator only ever needed these two values.
    """
    grounded = state.get("last_status") == ToolStatus.OK.value
    instruction = GENERATOR_PROMPT if grounded else GENERATOR_NO_CONTEXT_PROMPT
    logger.info(f"Generator node executing (grounded={grounded}).")

    prompt = f"{instruction}\n\nQUESTION:\n{_original_question(state)}\n\nTOOL OUTPUT:\n{_latest_tool_output(state)}"

    response = generator_llm.invoke([HumanMessage(content=prompt)])

    if not response.text.strip():
        # A blank completion would render as an empty answer with no
        # explanation. Surface the tool's own answer instead of nothing.
        logger.warning("Generator returned empty text; falling back to the raw tool payload.")
        fallback = _latest_tool_output(state) or "The agent produced no answer."
        try:
            payload = json.loads(fallback).get("payload") or {}
            fallback = payload.get("answer") or fallback
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"messages": [AIMessage(content=str(fallback))]}

    logger.info("Final answer generated.")
    return {"messages": [response]}


# --- Routing ---
def should_continue(state: AgentState) -> str:
    """Edge out of the manager: call a tool, or stop."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"Decision: call tool '{last_message.tool_calls[0]['name']}'.")
        return "call_tool"
    logger.info("Decision: manager returned no tool call; ending execution.")
    return "end"


def route_after_tool(state: AgentState) -> str:
    """Edge out of the tool executor: retry a different route, or synthesise.

    Thin adapter over `decide_after_tool`, which holds the policy and is unit
    tested independently of LangGraph and of any model.
    """
    decision = decide_after_tool(
        status=state.get("last_status") or ToolStatus.ERROR.value,
        hops=state.get("hops", 0),
        failed_tools=state.get("failed_tools", []),
        total_tools=len(tools),
        max_hops=settings.MAX_TOOL_HOPS,
    )
    logger.info(
        f"Post-tool routing: status={state.get('last_status')} hops={state.get('hops')} "
        f"burned={sorted(set(state.get('failed_tools', [])))} -> {decision}"
    )
    return decision


# --- Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_node)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("manager")

workflow.add_conditional_edges(
    "manager",
    should_continue,
    {"call_tool": "tool_executor", "end": END},
)
workflow.add_conditional_edges(
    "tool_executor",
    route_after_tool,
    {RETRY: "manager", GENERATE_ANSWER: "generator"},
)
workflow.add_edge("generator", END)

agent_graph = workflow.compile()
logger.info("Agent graph compiled successfully.")