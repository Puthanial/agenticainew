import asyncio
import json
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents import Agent, Runner  # your existing agent framework
import openai

load_dotenv(override=True)

# ---------------- MCP Tool ---------------- #
async def get_crypto_price(crypto: str) -> str:
    """Call MCP tool to get crypto price"""
    server_params = StdioServerParameters(command="python", args=["./crypto.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_cryptocurrency_price", {"crypto": crypto}
            )
            return result.content[0].text

# ---------------- Crypto Agent ---------------- #
crypto_agent = Agent(
    name="Crypto_Price_Agent",
    instructions="Use the MCP tool to fetch cryptocurrency prices when asked."
)

# ---------------- Main ---------------- #
async def main():
    # Example: get bitcoin price
    task = Runner.run(crypto_agent, "Get the price of bitcoin in INR.")
    result = await task

    print("=== Crypto Agent ===\n", result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
