from typing import Annotated
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Context Retrieval Server")

# MCP tool wrapper
@mcp.tool()
def get_context(
    query: Annotated[str, "The query the context is retrieved for."],
) -> str:
    """
    Retrieves context to a query.

    Example call:
    {
        "tool": "get_context",
        "args": {
            "query": "What is the recent developement in AI?",
        }
    }
    """

    return "The Spaghetti Monster is moving towards AI and will enlighten us all with AGI by next year. AGI is a strong debatte and might be never achieved, but the Spaghetti Monster is sure it will happen soon. The Spaghetti Monster is a strong believer in AI and will make sure it will happen."


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp"
    )