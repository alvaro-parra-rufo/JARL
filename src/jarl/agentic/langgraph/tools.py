"""LangChain tool adapters for agentic registry entries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool

from jarl.agentic.tools.specs import BoundTool, ToolEntry, ToolRegistry

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "bound_tool_from_structured",
    "bound_tools_from_registry",
    "build_langchain_tools",
    "structured_tool_from_entry",
]


def structured_tool_from_entry(
    entry: ToolEntry,
    handler: Callable[..., str],
) -> StructuredTool:
    """Bind a registry entry to a LangChain ``StructuredTool``.

    The Pydantic JSON schema is exposed as ``args_schema`` without pre-validating
    in LangChain. Callers that execute tools should pass ``wire_handler`` so
    malformed calls are still audited.
    """
    return StructuredTool.from_function(
        func=handler,
        name=entry.spec.name,
        description=entry.spec.description,
        args_schema=entry.request_cls.model_json_schema(),
    )


def bound_tool_from_structured(tool: BaseTool) -> BoundTool:
    """Dump the LLM-facing name, description, and ``tool_call_schema`` of ``tool``."""
    schema = tool.tool_call_schema
    if not isinstance(schema, dict):
        msg = f"StructuredTool {tool.name!r} tool_call_schema is not a dict."
        raise TypeError(msg)
    return BoundTool(
        name=tool.name,
        description=tool.description or "",
        schema=schema,
    )


def bound_tools_from_registry(registry: ToolRegistry) -> tuple[BoundTool, ...]:
    """Build catalog payloads with the same ``StructuredTool`` bind as the graph."""
    return tuple(
        bound_tool_from_structured(structured_tool_from_entry(entry, _unused_catalog_handler)) for entry in registry
    )


def build_langchain_tools(registry: ToolRegistry, workflow: AgenticWorkflow) -> list[BaseTool]:
    """Convert filtered registry entries into LangChain tools."""
    return [structured_tool_from_entry(entry, entry.build_handler(workflow)) for entry in registry]


def _unused_catalog_handler(*args: object, **kwargs: object) -> str:
    del args, kwargs
    return ""
