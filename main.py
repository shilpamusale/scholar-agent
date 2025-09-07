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
main.py: The primary entry point for the Scholar-Agent application.
"""

import argparse
import json
import os
import warnings
from typing import Any

from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# --- CRITICAL CONFIGURATION (RUNS BEFORE LOCAL IMPORTS) ---
# Set the log level BEFORE importing any local application modules that use logging.
os.environ["LOG_LEVEL"] = "WARNING"
# Suppress Deprecation Warnings for a clean demo.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# --- LOCAL IMPORTS (PLACED AFTER CONFIGURATION) ---
# noqa: E402 tells the linter to ignore the "import not at top of file" error.
# This is intentional and necessary for the logging configuration to work correctly.
from src.agent.graph import agent_graph  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

# Initialize logger now that the environment variable is set.
logger = setup_logging(__name__)


def get_tool_call_from_ai_message(msg: AIMessage) -> dict[str, Any] | None:
    """Helper to safely extract the first tool call from an AIMessage."""
    if not isinstance(msg, AIMessage) or not msg.tool_calls:
        return None
    return msg.tool_calls[0]


def main():
    """The main execution function for the agent CLI."""
    parser = argparse.ArgumentParser(description="Run the Scholar-Agent with a research query.")
    parser.add_argument("query", type=str, help="The research query to ask the agent.")
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel(
            f"[bold cyan]Query:[/bold cyan] {args.query}",
            title="[bold green] ScholarAgent Initialized[/bold green]",
            border_style="green",
        )
    )

    inputs = {"messages": [HumanMessage(content=args.query)]}
    final_state: dict[str, Any] = {}

    try:
        with console.status("[bold green]Agent is thinking...", spinner="dots") as status:
            for chunk in agent_graph.stream(inputs):
                for node_name, state_update in chunk.items():
                    if node_name == "manager":
                        tool_call = get_tool_call_from_ai_message(state_update["messages"][-1])
                        if tool_call:
                            tool_name = tool_call.get("name", "Unknown Tool")
                            status.update(f"Deciding to call tool: [bold]{tool_name}[/bold]...")
                            panel_content = Markdown(f"**Decision:** Call tool `{tool_name}`.")
                            console.print(
                                Panel(
                                    panel_content,
                                    title="[bold cyan] Agent Thought Process[/bold cyan]",
                                    border_style="dim",
                                )
                            )

                    elif node_name == "tool_executor":
                        last_message = state_update["messages"][-1]
                        if isinstance(last_message, ToolMessage):
                            try:
                                tool_output_dict = json.loads(last_message.content)

                                if "database_result" in tool_output_dict:
                                    tool_name = "Knowledge Graph Tool"
                                    cypher_query = tool_output_dict.get("cypher_query", "No query generated.")
                                    db_result = tool_output_dict.get("database_result", [])

                                    query_panel = Panel(
                                        Syntax(
                                            cypher_query,
                                            "cypher",
                                            theme="monokai",
                                            line_numbers=True,
                                        ),
                                        title="[bold]Generated Cypher Query[/bold]",
                                        border_style="green",
                                    )

                                    table = Table(
                                        title="Database Results",
                                        expand=True,
                                        border_style="green",
                                    )
                                    if db_result:
                                        headers = db_result[0].keys()
                                        for header in headers:
                                            table.add_column(header, style="cyan", no_wrap=False)
                                        for row in db_result:
                                            table.add_row(*[str(item) for item in row.values()])

                                    console.print(
                                        Panel(
                                            query_panel,
                                            title=f"[bold yellow] Tool Output: {tool_name}[/bold yellow]",
                                            border_style="dim",
                                        )
                                    )
                                    console.print(Panel(table, border_style="dim"))

                                elif "answer" in tool_output_dict:
                                    tool_name = "Advanced RAG Tool"
                                    console.print(
                                        Panel(
                                            Markdown(tool_output_dict["answer"]),
                                            title=f"[bold yellow] Tool Output: {tool_name}[/bold yellow]",
                                            border_style="dim",
                                        )
                                    )

                                else:
                                    output_str = f"```json\n{last_message.content}\n```"
                                    console.print(
                                        Panel(
                                            Markdown(output_str),
                                            title="[bold yellow] Tool Output: Generic Tool[/bold yellow]",
                                            border_style="dim",
                                        )
                                    )

                            except json.JSONDecodeError:
                                output_str = f"```\n{last_message.content}\n```"
                                console.print(
                                    Panel(
                                        Markdown(output_str),
                                        title="[bold yellow] Tool Output: Raw Text[/bold yellow]",
                                        border_style="dim",
                                    )
                                )

                            status.update("Tool executed. Synthesizing final answer...")

                    final_state = state_update

        if "messages" in final_state and final_state["messages"]:
            final_answer = final_state["messages"][-1].content
            console.print(
                Panel(
                    Markdown(final_answer),
                    title="[bold blue] Final Answer[/bold blue]",
                    border_style="blue",
                    expand=True,
                )
            )
        else:
            console.print(
                Panel(
                    "Agent finished without a final answer.",
                    title="[bold yellow] Warning[/bold yellow]",
                    border_style="yellow",
                )
            )

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
        console.print(
            Panel(
                (
                    f"An error occurred: {e}\nPlease check the application log"
                    " file `logs/scholar_agent.log` for details."
                ),
                title="[bold red] Error[/bold red]",
                border_style="red",
            )
        )


if __name__ == "__main__":
    main()
