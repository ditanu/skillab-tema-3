"""
Client de test pentru serverul MCP L6.

Implicit testeaza handshake + tools/list fara sa apeleze LLM/DB:
    python src/test_mcp_client.py

Pentru test end-to-end, dupa ce ai .env, DB si datele pornite:
    python src/test_mcp_client.py --call
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).parent / "analyst_mcp_server.py"


async def main(call_tools: bool) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"Connected to {init.serverInfo.name} v{init.serverInfo.version}")

            tools = (await session.list_tools()).tools
            print("Tools:")
            for tool in tools:
                required = tool.inputSchema.get("required", [])
                print(f"  - {tool.name} | required={required}")

            if not call_tools:
                return

            analyst_result = await session.call_tool(
                "data_analyst",
                {
                    "question": "Care sunt top 5 furnizori dupa valoare?",
                    "include_plan": True,
                    "include_preview": True,
                    "max_preview_rows": 5,
                },
            )
            print("\n[data_analyst]")
            print(f"isError={analyst_result.isError}")
            print(analyst_result.content[0].text)

            orchestrator_result = await session.call_tool(
                "orchestrator_rag",
                {
                    "query": "Care e totalul facturilor TechSoft?",
                    "include_rag_context": True,
                },
            )
            print("\n[orchestrator_rag]")
            print(f"isError={orchestrator_result.isError}")
            print(orchestrator_result.content[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testeaza serverul MCP L6.")
    parser.add_argument(
        "--call",
        action="store_true",
        help="Apeleaza ambele tool-uri; necesita LLM, DB si date disponibile.",
    )
    args = parser.parse_args()

    asyncio.run(main(call_tools=args.call))
