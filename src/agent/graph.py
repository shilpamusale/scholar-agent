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

This module defines the state, nodes, and edges for the multi-agent system of
the Scholar-Agent. It uses LangGraph to create a stateful, cyclical graph that
enables robust, tool-using behavior.

The architecture follows a standard agentic pattern:
1.  A "manager" node acts as the primary router, deciding whether to call a
    tool or respond directly.
2.  A "tool_executor" node is responsible for invoking the chosen tool with the
    correct arguments.
3.  A "generator" node synthesizes the final response for the user based on the
    tool's output.
4.  Conditional edges route the flow between these nodes based on the state of
    the conversation.

The final compiled graph, `agent_graph`, is a runnable object that encapsulates
the entire reasoning process of the agent.
"""

# src/agent/graph.py

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

import configs.settings as settings
from configs.prompts import GENERATOR_PROMPT, MANAGER_PROMPT
from src.agent.tools import get_tools
from src.utils.logging_config import setup_logging

# logger = setup_logging(__name__, "agent_graph")
logger = setup_logging(__name__)

# --- LLM and Tool Initialization (Done Once) ---
tools = get_tools()
manager_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    google_api_key=settings.get_google_api_key(),
).bind_tools(tools)
generator_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    google_api_key=settings.get_google_api_key(),
)


# --- Agent Definition ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def manager_node(state: AgentState) -> dict:
    logger.info("Manager node executing.")
    messages_with_prompt = [HumanMessage(content=MANAGER_PROMPT)] + state["messages"]
    response = manager_llm.invoke(messages_with_prompt)
    logger.info(f"Manager response: {response}")
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Executes the tool called by the manager."""
    logger.info("Tool node executing.")
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    logger.info(f"Executing {tool_name} with args: {tool_args}")

    tool_map = {tool.name: tool for tool in tools}
    tool_to_call = tool_map.get(tool_name)

    if tool_to_call:
        # The 'args' are now a dictionary with a 'question' key,
        query = tool_args.get("question")
        if query is None:
            # Fallback for safety, though schema should prevent this
            query = next(iter(tool_args.values()), None)

        if query:
            response = tool_to_call.invoke(query)
            logger.info("Tool execution finished.")
            return {"messages": [ToolMessage(content=str(response), tool_call_id=tool_call["id"])]}
        else:
            logger.warning("Tool called with no valid query argument.")
            return {
                "messages": [
                    ToolMessage(
                        content="Error: Tool called with no query.",
                        tool_call_id=tool_call["id"],
                    )
                ]
            }
    else:
        logger.warning(f"Tool '{tool_name}' not found.")
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found.",
                    tool_call_id=tool_call["id"],
                )
            ]
        }


def generator_node(state: AgentState) -> dict:
    logger.info("Generator node executing.")
    messages_with_prompt = [HumanMessage(content=GENERATOR_PROMPT)] + state["messages"]
    response = generator_llm.invoke(messages_with_prompt)
    logger.info("Final answer generated.")
    return {"messages": [response]}


# --- Graph Construction (no changes here) ---
def should_continue(state: AgentState) -> str:
    logger.info("Router checking the last message.")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Decision: Call tools.")
        return "call_tool"
    if isinstance(last_message, ToolMessage):
        logger.info("Decision: Generate final answer.")
        return "generate_answer"
    logger.info("Decision: End execution.")
    return "end"


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
workflow.add_edge("tool_executor", "generator")
workflow.add_edge("generator", END)

agent_graph = workflow.compile()
logger.info("Agent graph compiled successfully.")
