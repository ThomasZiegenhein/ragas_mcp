import asyncio
import pprint
import pytest
import requests
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

import sys
sys.path.append("..")  # Adjust the path to import mcp

POSSIBLE_MCP_METRICS_URLS = [
    "http://mcp_workflow_server:8000/mcp",
    "http://localhost:8001/mcp"
]

def find_working_mcp_url(urls):
    for url in urls:
        try:
            resp = requests.get(url.replace("8000/mcp", "8000"))
            if resp.status_code < 500:
                return url
        except Exception:
            continue
    raise RuntimeError("No working MCP metrics server URL found.")

MCP_URL = find_working_mcp_url(POSSIBLE_MCP_METRICS_URLS)

async def call_mcp(tool_name: str, args: dict):
    """
    Sends a request to the MCP server and returns the result.
    Raises an exception if the server returns an error.
    """
    async with streamablehttp_client(MCP_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            tool_result = await session.call_tool(tool_name, args)
            return tool_result

def test_mcp_server_running():
    """Assert that the MCP server is running and reachable at MCP_URL."""
    try:
        response = requests.get(MCP_URL)
        # Accept any 2xx or 4xx response as "running" (since /mcp may not be GET)
        assert response.status_code < 500
    except Exception as e:
        pytest.fail(f"MCP server is not running or not reachable at {MCP_URL}: {e}")

@pytest.mark.asyncio
async def test_evaluate_question_answer_with_context_workflow():
    llm = "gpt-4o-mini"
    embedding_model = "text-embedding-3-small"
    framework = "ragas"

    # 1) Faithfulness
    faith_args = {
        "user_input": "What city is the Eiffel Tower located in?",
        "response": "The Eiffel Tower is located in Berlin.",
        "reference_answer": "The Eiffel Tower is located in Paris, France.",
        "retrieved_contexts": "The Eiffel Tower is located in Paris, France.",
        "eval_framework": framework,
        "llm": llm,
        "embedding_model": embedding_model,
    }
    scores= await call_mcp("evaluate_question_answer_with_context_workflow", faith_args)
    assert scores is not None
    return scores

if __name__ == "__main__":
    pprint.pprint([x.text for x  in asyncio.run(test_evaluate_question_answer_with_context_workflow()).content])
