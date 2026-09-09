"""Stdio FastMCP server for JARL tools."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from jarl.agentic.errors import ExperimentNotFoundError
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.specs import ToolRegistry
from jarl.agentic.workflow import AgenticWorkflow
from jarl.logging import redirect_console_to_stderr
from jarl.mcp.bridge import register_tools

__all__ = [
    "JARL_MCP_EXPERIMENT_DIR_ENV",
    "SERVER_INSTRUCTIONS",
    "configure_stderr_logging",
    "create_server",
    "main",
    "run",
    "workflow_from_env",
]

JARL_MCP_EXPERIMENT_DIR_ENV = "JARL_MCP_EXPERIMENT_DIR"
"""Environment variable with the experiment directory bound to this process."""

SERVER_INSTRUCTIONS = (
    "JARL experiment tools. Set "
    f"{JARL_MCP_EXPERIMENT_DIR_ENV} to an experiment directory that contains "
    "experiment.json. Use the catalog tools to inspect and mutate the graph "
    "and to run training."
)
"""MCP server instructions returned during initialization."""

_LOGGER_NAME = "jarl.mcp"
"""Logger name for this process; handlers write to stderr only."""


def configure_stderr_logging() -> None:
    """Send process logs to stderr so stdout stays JSON-RPC."""
    redirect_console_to_stderr()
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def workflow_from_env() -> AgenticWorkflow:
    """Load the experiment bound by `JARL_MCP_EXPERIMENT_DIR`.

    Raises:
        ExperimentNotFoundError: If the environment variable is missing or the
            directory has no experiment manifest.
    """
    raw = os.environ.get(JARL_MCP_EXPERIMENT_DIR_ENV, "").strip()
    if not raw:
        msg = f"{JARL_MCP_EXPERIMENT_DIR_ENV} is not set."
        raise ExperimentNotFoundError(msg)
    return AgenticWorkflow.from_experiment(Path(raw).expanduser())


def create_server(
    *,
    workflow: AgenticWorkflow | None = None,
    registry: ToolRegistry | None = None,
) -> FastMCP:
    """Build the JARL FastMCP server and register the agentic catalog.

    Logs go to stderr so stdout stays JSON-RPC. When `workflow` is omitted,
    the process loads `JARL_MCP_EXPERIMENT_DIR` once.

    Args:
        workflow: Experiment workflow. Loaded from the environment when omitted.
        registry: Tool catalog. Defaults to `REGISTRY`.
    """
    configure_stderr_logging()
    active_workflow = workflow if workflow is not None else workflow_from_env()
    catalog = REGISTRY if registry is None else registry
    server = FastMCP(
        name="jarl",
        instructions=SERVER_INSTRUCTIONS,
        log_level="INFO",
    )
    register_tools(server, catalog, active_workflow)
    return server


def run(server: FastMCP | None = None) -> None:
    """Serve MCP over stdin/stdout.

    Args:
        server: Server to run. When omitted, `create_server` builds one.
    """
    active = create_server() if server is None else server
    active.run(transport="stdio")


def main() -> int:
    """Run the JARL MCP server over stdio.

    Returns:
        Process exit code. `2` when the experiment directory is missing.
    """
    configure_stderr_logging()
    try:
        run()
    except ExperimentNotFoundError as exc:
        logging.getLogger(_LOGGER_NAME).error("%s", exc)
        return 2
    return 0
