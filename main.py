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
import logging
import os
import warnings
from typing import Any

from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.messages import AIMessage, ToolMessage
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

# The google-genai client logs an advisory about automatic function calling on
# every request. It is informational, not a problem with this application, and
# it is emitted through logging rather than warnings, so a warnings filter does
# not reach it. Logger name confirmed in google/genai/_extra_utils.py.
logging.getLogger("google_genai").setLevel(logging.ERROR)


# --- LOCAL IMPORTS (PLACED AFTER CONFIGURATION) ---
# noqa: E402 tells the linter to ignore the "import not at top of file" error.
# This is intentional and necessary for the logging configuration to work correctly.
from src.agent.graph import agent_graph, initial_state  # noqa: E402
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

    inputs = initial_state(args.query)
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
                                envelope = json.loads(last_message.text)
                            except json.JSONDecodeError:
                                envelope = {
                                    "tool": "unknown",
                                    "status": "error",
                                    "detail": "Tool output was not a valid envelope.",
                                    "payload": {"raw": last_message.text},
                                }

                            tool_name = envelope.get("tool", "unknown tool")
                            tool_status = envelope.get("status", "unknown")
                            payload = envelope.get("payload") or {}
                            border = "green" if tool_status == "ok" else "yellow"
                            header = f"[bold yellow] Tool Output: {tool_name} ({tool_status})[/bold yellow]"

                            if "cypher_query" in payload:
                                console.print(
                                    Panel(
                                        Syntax(
                                            payload["cypher_query"],
                                            "cypher",
                                            theme="monokai",
                                            line_numbers=True,
                                        ),
                                        title=header,
                                        border_style=border,
                                    )
                                )
                                records = payload.get("database_result") or []
                                if records:
                                    table = Table(title="Database Results", expand=True, border_style=border)
                                    for header_name in records[0].keys():
                                        table.add_column(header_name, style="cyan", no_wrap=False)
                                    for row in records:
                                        table.add_row(*[str(item) for item in row.values()])
                                    console.print(Panel(table, border_style="dim"))

                            elif "answer" in payload:
                                console.print(
                                    Panel(
                                        Markdown(payload["answer"]),
                                        title=header,
                                        border_style=border,
                                    )
                                )
                                sources = payload.get("sources") or []
                                if sources:
                                    source_lines = "\n".join(
                                        f"- `{src.get('source', 'unknown')}` p.{src.get('page', '?')}"
                                        for src in sources
                                    )
                                    console.print(
                                        Panel(
                                            Markdown(source_lines),
                                            title="[bold]Retrieved Sources[/bold]",
                                            border_style="dim",
                                        )
                                    )

                            else:
                                console.print(
                                    Panel(
                                        Markdown(f"```json\n{json.dumps(payload, indent=2)}\n```"),
                                        title=header,
                                        border_style=border,
                                    )
                                )

                            if tool_status == "ok":
                                status.update("Route produced grounding. Synthesizing final answer...")
                            else:
                                console.print(
                                    Panel(
                                        Markdown(
                                            f"**{tool_name}** returned status `{tool_status}`.\n\n"
                                            f"{envelope.get('detail', '')}"
                                        ),
                                        title="[bold red] Route Failed - Falling Back[/bold red]",
                                        border_style="red",
                                    )
                                )
                                status.update("Route failed. Trying an alternate route...")

                    final_state = state_update

        if "messages" in final_state and final_state["messages"]:
            # `.content` may be a list of typed content blocks; `.text`
            # flattens it to the string Markdown expects.
            final_answer = final_state["messages"][-1].text
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
