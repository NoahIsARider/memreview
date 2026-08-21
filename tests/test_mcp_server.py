#!/usr/bin/env python3
"""End-to-end test for the memreview MCP server (stdio, isolated HOME)."""
import asyncio
import json
import os
import shutil
import sys

TEST_HOME = "/tmp/memreview-mcp-test"
shutil.rmtree(TEST_HOME, ignore_errors=True)
os.makedirs(TEST_HOME)

ENV = os.environ.copy()
ENV["MEMREVIEW_HOME"] = TEST_HOME

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_CMD = [sys.executable, "-m", "memreview.mcp_server"]


async def main():
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:], env=ENV)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"✅ {len(names)} tools: {', '.join(names)}")
            assert len(names) == 13, f"expected 13 tools, got {len(names)}"

            async def call(name, **kwargs):
                res = await session.call_tool(name, arguments=kwargs or {})
                text = res.content[0].text if res.content else "(empty)"
                print(f"\n▶ {name}({kwargs})")
                print(text[:500])
                return text

            await call("status")

            r = await call("srs_add", category="test", front="MCP test item",
                           back="this is a test", example="from pytest")
            assert "test-001" in r

            await call("srs_due")
            await call("srs_stats")

            await call("memory_daily", content="today I built the MCP server")
            r = await call("memory_read", path=f"{__import__('datetime').date.today()}.md")
            assert "MCP server" in r

            await call("memory_write", filename="notes/test.md", content="# hello memory")
            r = await call("memory_read", path="notes/test.md")
            assert "hello memory" in r

            # path traversal must be rejected (server returns isError)
            res = await session.call_tool("memory_read", arguments={"path": "../../etc/passwd"})
            if not getattr(res, "isError", False):
                print("❌ traversal NOT blocked!")
                sys.exit(1)
            print("✅ traversal blocked (server returned error)")

            await call("context_save", task="testing MCP server", kind="task-switch")
            await call("context_restore")
            await call("contexts_list")

            await call("index_rebuild")
            r = await call("memory_search", query="MCP server", n=3)
            assert "MCP server" in r, f"search missed content: {r}"

            r = await call("srs_review", item_id="test-001", correct=True)
            assert "test-001" in r

            print("\n🎉 ALL MCP SERVER TESTS PASSED")


asyncio.run(main())
