#!/usr/bin/env python3
"""
agent.py

An interactive command-line AI assistant that uses multiple MCP servers via SSE transport.
"""

import asyncio
import pprint
from dotenv import load_dotenv
load_dotenv("/workspaces/ragas_mcp/.env")  # Load environment variables from .env file

import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI

# Import the MultiServerMCPClient from the MCP adapters package
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent


def setup_agent(server_configs: Dict[str, dict]) -> Any:
    """
    Initialize and return an agent with tools loaded from multiple MCP servers.
    """
    # Create the MultiServerMCPClient with configured servers
    mcp_client = MultiServerMCPClient(connections=server_configs)
    tools = asyncio.run(mcp_client.get_tools())

    # Define system prompt for the agent
    system_prompt = (
        "You are an advanced AI assistant. "
        "Use the context server to recieve context to your questions"
        "After you created an answer, use the workflow_server to execute the answer."
        "The workflow server will return metrics that will allow you to evaluate your answer."
        "Return your answer and the metrics in a single response."
    )

    # Instantiate the LLM with deterministic settings
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_react_agent(llm, tools, prompt=system_prompt)

    return agent


def main():
    """
    Run an interactive loop prompting the user for input and returning agent responses.
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define MCP servers with SSE transport protocol
    server_configs = {
        "evaluation_server": {
            "url": "http://localhost:8001/mcp",
            "transport": "streamable_http",
        },
        "context_server": {
            "url": "http://localhost:8002/mcp",
            "transport": "streamable_http",
        },
        "reference_data_server": {
            "url": "http://localhost:8003/mcp",
            "transport": "streamable_http",
        },
    }

    # Initialize the agent with the specified MCP servers
    agent = setup_agent(server_configs)
    system_prompt = (
        "You are an advanced AI assistant. "
        "Use the context server to recieve context to your questions"
        "After you created an answer, use the reference data server to get the ground truth."
        "Finally, evaluate your answer with the ground truth and context using the evaluation_server."
        "Return your answer and the metrics in a single response."
    )

    try:
        print(' '*10 +"#"*115)
        print(' '*25 +"Welcome to the Agent Evaluation Demo!")
        print(' '*25 +"The Agent will evaluate itself via connecting to a MCP evaluation service server.")
        print(' '*25 +"The Agent is connected to the following MCP servers:")
        for name, config in server_configs.items():
            print(' '*30 + f"{name}: {config['url']} (Transport: {config['transport']})")
        print(' '*10 + "#"*115)
        while True:
            print("Type your question or 'exit' to quit:")
            user_input = input("What is the current status of AI?")
            if not user_input.strip():
                user_input = "What is the current status of AI"
            print("Your question is: {}".format(user_input))
            if user_input.strip().lower() in ("exit", "quit"):
                logger.info("Shutting down assistant.")
                break
            response = asyncio.run(agent.ainvoke({"messages": [SystemMessage(content=system_prompt),
                                                               HumanMessage(content=user_input)]}))
            
            pprint.pprint(response, indent=2)
            print('\n\n' + ' '*10 + "#"*25 + "Final Agent step:" + "#"*25 )
            print(response["messages"][-1].content)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")


if __name__ == "__main__":
    main()
