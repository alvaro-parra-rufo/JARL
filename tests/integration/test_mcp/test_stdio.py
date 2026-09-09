"""Stdio handshake and `tools/list` against the live JARL MCP process."""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from jarl.agentic.tools.registry import REGISTRY
from jarl.mcp.server import JARL_MCP_EXPERIMENT_DIR_ENV, SERVER_INSTRUCTIONS


def _stdio_params(exp_dir: Path) -> StdioServerParameters:
    """Build stdio spawn parameters for `python -m jarl.mcp`."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "jarl.mcp"],
        env={
            JARL_MCP_EXPERIMENT_DIR_ENV: str(exp_dir),
            "PYTHONUNBUFFERED": "1",
            "JAX_PLATFORMS": "cpu",
        },
    )


class TestMcpStdio:
    def test_handshake_and_tools_list_match_registry(self, prepared_experiment: Path) -> None:
        """Initialize over stdio and list the same names as `REGISTRY`."""

        async def _run() -> tuple[str | None, set[str]]:
            async with (
                stdio_client(_stdio_params(prepared_experiment)) as (read, write),
                ClientSession(read, write) as session,
            ):
                result = await session.initialize()
                listed = await session.list_tools()
                return result.instructions, {tool.name for tool in listed.tools}

        instructions, names = anyio.run(_run)

        assert instructions == SERVER_INSTRUCTIONS
        assert names == set(REGISTRY.names())
