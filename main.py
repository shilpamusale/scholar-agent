# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of the agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
main.py: The main entry point for the Scholar-Agent application.

This script serves as the command-line interface for interacting with the
multi-agent system. It takes a user's research query as input, invokes the
compiled agent graph, and streams the final, synthesized answer back to the
console.

Usage:
    python main.py "Your research question here"
"""

import argparse

from langchain_core.messages import HumanMessage

from src.agent.graph import agent_graph
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__, "main")


def main():
    """
    The main execution function for the Scholar-Agent.
    """
    parser = argparse.ArgumentParser(description="Scholar-Agent CLI")
    parser.add_argument("query", type=str, help="The research query to ask the agent.")
    args = parser.parse_args()

    logger.info(f"Starting agent with query: {args.query}")

    final_state = None
    try:
        # Define the input for the agent graph (CORRECTED LINE)
        inputs = {"messages": [HumanMessage(content=args.query)]}

        # Stream the execution of the agent graph
        for chunk in agent_graph.stream(inputs):
            # The stream yields a dictionary
            # with the node name as the key
            # and the node's output as the value.
            # We print each step for visibility.
            print("---")
            print(chunk)
            print("---")
            final_state = chunk

        # The final answer is in the 'generator'
        # node's output from the last chunk
        if final_state and "generator" in final_state:
            final_answer = final_state["generator"]["messages"][-1].content
            print("\n~~~~~~~~~~~\nFinal Answer:\n\n", final_answer, "\n~~~~~~~~~~~\n")
        else:
            print("\n~~~~~~~~~~~")
            print("\nThe agent finished, but no final answer was generated.")
            print("\n~~~~~~~~~~~\n")
            print("Final state:", final_state)

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
        print("\nAn error occurred. Please check the logs for details.")


if __name__ == "__main__":
    main()
