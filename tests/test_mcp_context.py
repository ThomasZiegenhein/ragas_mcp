import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import requests

MCP_URL = "http://localhost:8002/mcp"

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

def test_get_context():
    payload = {
        "tool_name": "get_context",
        "args": {
            "query": "What is the recent development in AI?"
        }
    }
    # Properly run the async function from sync context
    response = asyncio.run(call_mcp(**payload))
    print(response)
    # Since response is likely a string or dict, adjust assertions accordingly
    assert "Spaghetti Monster" in str(response)

if __name__ == "__main__":
    test_get_context()
    print("Test passed.")