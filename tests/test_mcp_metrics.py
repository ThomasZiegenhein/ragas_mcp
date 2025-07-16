import asyncio
import pytest
import requests
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

import sys

sys.path.append("..")  # Adjust the path to import mcp

MCP_URL = "http://localhost:8000/mcp"

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
async def test_ragas():
    llm = "gpt-4o-mini"
    embedding_model = "text-embedding-3-small"
    framework = "ragas"

    # 1) Faithfulness
    faith_args = {
        "user_input": "What city is the Eiffel Tower located in?",
        "response": "The Eiffel Tower is located in Berlin.",
        "retrieved_contexts": "The Eiffel Tower is located in Paris, France.",
        "eval_framework": framework,
        "llm": llm
    }
    faith_score = await call_mcp("calculate_faithfulness", faith_args)
    print(faith_score)
    assert faith_score is not None


    # 2) Answer Relevancy
    relevancy_args = {
        "user_input": "Why is the sky blue?",
        "response": "Because of Rayleigh scattering.",
        "eval_framework": framework,
        "llm": llm,
        "embedding_model": embedding_model,
        "strictness": 3
    }
    relevancy_score = await call_mcp("calculate_answer_relevancy", relevancy_args)
    print(relevancy_score)
    assert relevancy_score is not None


    # 3) Context Precision
    precision_args = {
        "user_input": "Define photosynthesis.",
        "response": "Photosynthesis is the process by which plants convert light into chemical energy.",
        "retrieved_contexts": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
        "eval_framework": framework,
        "llm": llm
    }
    precision_score = await call_mcp("calculate_context_precision", precision_args)
    print(precision_score)
    assert precision_score is not None

    

    # 4) Context Recall
    recall_args = {
        "user_input": "Explain evolution.",
        "retrieved_contexts": "Evolution is the change in the heritable traits of biological populations over successive generations.",
        "reference_answer": "Evolution is the process by which different kinds of living organisms are thought to have developed and diversified from earlier forms during the history of the earth.",
        "eval_framework": framework,
        "llm": llm
    }
    recall_score = await call_mcp("calculate_context_recall", recall_args)
    print(recall_score)
    assert recall_score  is not None


    # 5) Answer Correctness
    correctness_args = {
        "user_input": "What is gravity?",
        "response": "Gravity is the force that attracts two masses toward each other.",
        "reference_answer": "Gravity is the natural phenomenon by which all things with mass or energy are brought toward one another.",
        "eval_framework": framework,
        "llm": llm,
        "embedding_model": embedding_model,
    }
    correctness_score = await call_mcp("calculate_answer_correctness", correctness_args)
    print(correctness_score)
    assert correctness_score  is not None
    

if __name__ == "__main__":
    asyncio.run(test_ragas())

