import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://127.0.0.1:8001/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # Get all tools from the server
            tools = await session.list_tools()
            print(f"list_tools() command: \n {tools}")
            

if __name__ == "__main__":
    server_configs = {
        "LLM Evaluation Workflow Server": {
            "url": "http://127.0.0.1:8001/mcp",
            "transport": "streamable_http",
        }
    }
    mcp_client = MultiServerMCPClient(server_configs)
    tools = asyncio.run(mcp_client.get_tools())
    #asyncio.run(main())

