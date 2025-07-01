from pathlib import Path
import sys

import requests
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP
from typing import Annotated
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from mcp_metric_server.my_llms import SUPPORTED_LLMS

# Initialize MCP server
mcp = FastMCP("LLM Evaluation Workflow Server")

SUPPORTED_EVAL_FRAMEWORKS = ["ragas"]

POSSIBLE_MCP_METRICS_URLS = [
    "http://mcp_metric_server:8000/mcp",
    "http://localhost:8000/mcp"
]

def find_working_mcp_url(urls):
    for url in urls:
        try:
            resp = requests.get(url.replace("8000/mcp", "8000"))  # Try root or adjust as needed
            if resp.status_code < 500:
                return url
        except Exception:
            continue
    raise RuntimeError("No working MCP metrics server URL found.")

CONNECTION_MCP_METRICS = find_working_mcp_url(POSSIBLE_MCP_METRICS_URLS)

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If already in an event loop, use create_task and gather result
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)

async def call_mcp(tool_name: str, args: dict):
    """
    Sends a request to the MCP server and returns the result.
    Raises an exception if the server returns an error.
    """
    async with streamablehttp_client(CONNECTION_MCP_METRICS) as (
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

# MCP tool wrapper
@mcp.tool()
def evaluate_question_answer_with_context_workflow(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    reference_answer: Annotated[str, "Optionally if available, the correct answer to the user input. If not available use an empty string."],
    retrieved_contexts: Annotated[str, "The retrieved context used to support the answer."],
    eval_framework: Annotated[str, "The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, "Identify the LLM-as-a-Judge, supported: {}".format(SUPPORTED_LLMS)],
    embedding_model: Annotated[str, "The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}."]
) -> float:
    """
    Calculates the scores for a question-answering task with context.

    This tool evaluates whether the generated `response` is grounded in the provided `context`, based on the original `user_input`.

    A dictionay with the calculated scores for the evaluation.

    This tool is useful to evaluate the quality of an agent in a question-answering task, especially when the agent uses context to generate its response.

    Example call:
    {
        "tool": "calculate_qa_with_context_workflow",
        "args": {
            "user_input": "What city is the Eiffel Tower located in?",
            "response": "The Eiffel Tower is located in Berlin.",
            "context": "The Eiffel Tower is located in Paris, France."
            "reference_answer": "The Eiffel Tower is located in Paris, France.",
            "eval_framework": "ragas",
            "llm": "gpt-4o-mini"
        }
    }
    """

    # Gather the metrics in our workflow and their settings
    jobs= [
        (
            "calculate_faithfulness",
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "eval_framework": eval_framework,
                "llm": llm
            }
        ),
        (
            "calculate_answer_relevancy",
            {
                "user_input": user_input,
                "response": response,
                "eval_framework": eval_framework,
                "llm": llm,
                "embedding_model": embedding_model,
                "strictness": 3
            }
        ),
        (
            "calculate_context_precision",
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "eval_framework": eval_framework,
                "llm": llm,
            }
        ),
        (
            "calculate_context_recall",
            {
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference_answer": reference_answer,
                "eval_framework": eval_framework,
                "llm": llm,
            }
        ),
        (
            "calculate_answer_correctness",
            {
                "user_input": user_input,
                "response" : response,
                "reference_answer": reference_answer,
                "eval_framework": eval_framework,
                "llm": llm,
                "embedding_model": embedding_model,
            }
        )
    ]

    async def run_jobs():
        tasks = [
            call_mcp(tool, args)
            for tool, args in jobs
        ]
        results = await asyncio.gather(*tasks)
        return {name: result for (name, _), result in zip(jobs, results)}
    
    return run_async(run_jobs())


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp"
    )