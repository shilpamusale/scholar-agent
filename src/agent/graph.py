# src/agent/graph.py

import operator

# from typing import Annotated, List, TypedDict
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

import configs.settings as settings
from src.agent.tools import get_tools
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "agent_graph")


# --- Agent Definition ---


# 1. Define the state
class AgentState(TypedDict):
    """The state of the graph, passed between nodes."""

    messages: Annotated[list[BaseMessage], operator.add]


# 2. Define the nodes
def manager_node(state: AgentState) -> dict:
    """The manager node decides the next action."""
    logger.info("Manager node executing.")
    system_prompt = (
        "You are a research manager. Your goal is to answer the user's query "
        "by calling the `research_paper_search` tool. Do not try to answer "
        "the question from your own knowledge. Always call the tool."
    )
    messages_with_prompt = [HumanMessage(content=system_prompt)] + state["messages"]

    tools = get_tools()
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME,
        temperature=0,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    ).bind_tools(tools)

    response = llm.invoke(messages_with_prompt)
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

    # Create a dispatcher to look up the tool by name
    tools = get_tools()
    tool_map = {tool.name: tool for tool in tools}
    tool_to_call = tool_map.get(tool_name)

    if tool_to_call:
        # The tool expects a single query string,
        # but Gemini passes it as {'__arg1': '...'}
        # or sometimes {'query': '...'}.
        # This handles both cases.
        query = next(iter(tool_args.values()))
        response = tool_to_call.invoke(query)
        logger.info("Tool execution finished.")
        return {
            "messages": [
                ToolMessage(content=str(response), tool_call_id=tool_call["id"])
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
    """Synthesizes the final answer after the tool has been called."""
    logger.info("Generator node executing.")
    # Refined prompt for higher quality synthesis
    system_prompt = (
        "You are an expert research assistant. "
        "Your task is to synthesize a clear and "
        "concise answer to the user's question based "
        "*only* on the provided context. "
        "Structure your answer logically. "
        "If the context contains multiple points, "
        "synthesize them into a coherent response. "
        "Do not add any information or "
        "opinions that are not explicitly stated in the context."
    )
    messages_with_prompt = [HumanMessage(content=system_prompt)] + state["messages"]

    generator_llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME,
        temperature=0,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    )
    response = generator_llm.invoke(messages_with_prompt)
    logger.info("Final answer generated.")
    return {"messages": [response]}


# 3. Define the conditional router
def should_continue(state: AgentState) -> str:
    """The router for our graph."""
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


# 4. Construct the graph
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
