"""Default tool registry for ``jarl.agentic``."""

from __future__ import annotations

from jarl.agentic.tools.specs import ToolRegistry

_TOOLS_PACKAGE = "jarl.agentic.tools"

REGISTRY: ToolRegistry = ToolRegistry.from_module(_TOOLS_PACKAGE)
"""Discovered project tool catalog under ``jarl.agentic.tools``."""

__all__ = ["REGISTRY"]
