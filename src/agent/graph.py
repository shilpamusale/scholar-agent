# src/agent/graph.py

import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# Corrected import name
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

import configs.settings as settings
from src.agent.tools import get_tools
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "agent_graph")

# Initialize the tools by calling the function
tools = get_tools()

# Bind the tools to the LLM that will act as our Manager
# Corrected class name
manager_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL_NAME, max_output_tokens=settings.MAX_OUTPUT_TOKENS
).bind_tools(tools)


# Define the state of our graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# --- Node Definitions ---


def manager_node(state: AgentState) -> dict:
    logger.info("Manager node executing.")
    system_prompt = (
        "You are a research manager. Your goal is to answer the user's query "
        "by calling the `research_paper_search` tool. Do not try to answer "
        "the question from your own knowledge. Always call the tool."
    )
    messages_with_prompt = [HumanMessage(content=system_prompt), state["messages"][-1]]
    response = manager_llm.invoke(messages_with_prompt)
    logger.info(f"Manager response: {response}")
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    logger.info("Tool node executing.")
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    logger.info(f"Executing {tool_name} with args: {tool_args}")
    if tool_name == "research_paper_search":
        query = next(iter(tool_args.values()))
        response = tools[0].invoke(query)
        logger.info("Tool execution finished.")
        return {
            "messages": [
                ToolMessage(content=str(response), tool_call_id=tool_call["id"])
            ]
        }
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def generator_node(state: AgentState) -> dict:
    logger.info("Generator node executing.")
    system_prompt = (
        "You are a helpful research assistant. "
        "Synthesize a concise answer to the user's "
        "question based ONLY on the provided "
        "context from the tools. "
        "Do not add any information that "
        "is not present in the context."
    )
    messages_with_prompt = [HumanMessage(content=system_prompt)] + state["messages"]
    # Corrected class name
    generator_llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME, max_output_tokens=settings.MAX_OUTPUT_TOKENS
    )
    response = generator_llm.invoke(messages_with_prompt)
    logger.info("Final answer generated.")
    return {"messages": [response]}


# --- Conditional Edge Logic ---


def should_continue(state: AgentState) -> str:
    logger.info("Router checking the last message...")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Decision: Call tools.")
        return "call_tool"
    if isinstance(last_message, ToolMessage):
        logger.info("Decision: Generate final answer.")
        return "generate_answer"
    logger.info("Decision: End execution.")
    return "end"


# --- Graph Definition ---

workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_node)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("manager")

workflow.add_conditional_edges(
    "manager",
    should_continue,
    {"call_tool": "tool_executor", "end": END, "generate_answer": "generator"},
)
workflow.add_edge("tool_executor", "generator")
workflow.add_edge("generator", END)

agent_graph = workflow.compile()
logger.info("Agent graph compiled successfully.")
