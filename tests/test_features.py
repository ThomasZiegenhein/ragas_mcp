import os

import pytest
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import requests

# Feature 1: MCP Metrics API endpoints (basic liveness and tool list)
def test_mcp_metric_server_running():
    url = os.getenv("MCP_URL", "http://localhost:8000/mcp")
    resp = requests.get(url)
    assert resp.status_code < 500


# Feature 2: calculate_faithfulness tool (and others) using MCP client
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")

async def call_mcp(tool_name: str, args: dict):
    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tool_result = await session.call_tool(tool_name, args)
            return tool_result

@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", [
    ("calculate_faithfulness", {
        "user_input": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "retrieved_contexts": "Paris is the capital of France.",
        "eval_framework": "ragas",
        "llm": "gpt-4o-mini"
    }),
    ("calculate_answer_relevancy", {
        "user_input": "Why is the sky blue?",
        "response": "Because of Rayleigh scattering.",
        "eval_framework": "ragas",
        "llm": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small"
    }),
    ("calculate_context_precision", {
        "user_input": "Define photosynthesis.",
        "response": "Photosynthesis is the process...",
        "retrieved_contexts": "Photosynthesis is the process...",
        "eval_framework": "ragas",
        "llm": "gpt-4o-mini"
    }),
    ("calculate_context_recall", {
        "user_input": "Explain evolution.",
        "retrieved_contexts": "Evolution is ...",
        "reference_answer": "Evolution is ...",
        "eval_framework": "ragas",
        "llm": "gpt-4o-mini"
    }),
    ("calculate_answer_correctness", {
        "user_input": "What is gravity?",
        "response": "Gravity is the force that attracts objects...",
        "reference_answer": "Gravity is the force of attraction...",
        "eval_framework": "ragas",
        "llm": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small"
    })
])
async def test_metric_tool(tool, args):
    result = await call_mcp(tool, args)
    assert result is not None
    assert isinstance(result, float)

# Feature 3: LLM/embedding provider selection (env var)
def test_llm_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src.my_llms import detect_provider
    assert detect_provider() == "openai"

# Feature 4: Embedding provider selection (env var)
def test_embedding_provider_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    from src.my_llms import detect_embedding_provider
    assert detect_embedding_provider() == "openai"

# Feature 5: Docker secret/env config (simulate API key)
def test_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    from src.my_llms import load_api_key
    assert load_api_key("openai") == "dummy"

# Feature 6: Protocol override env vars
def test_openai_protocol_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.example.com")
    from src.my_llms import get_llm
    llm = get_llm()
    assert hasattr(llm, "model_name") or hasattr(llm, "model")

# Feature 7: Agent/Workflow integration (smoke test)
def test_agent_example_exists():
    assert os.path.exists("example/agent/agent.py")

# Feature 8: Docker Compose file exists
def test_docker_compose_exists():
    assert os.path.exists("example/docker_composer.yml")
    assert os.path.exists("src/requirements.txt")
