"""Registry of subagent LangGraph builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from jarl.agentic.tools.context import ToolContext
    from jarl.agentic.tools.specs import ToolRegistry

__all__ = ["SubgraphBuilder", "get_subgraph_builder", "subgraph_graph_ids"]

SubgraphBuilder = Callable[["ToolContext", "ToolRegistry", BaseChatModel | None], CompiledStateGraph]


_BUILDERS: dict[str, SubgraphBuilder] = {}


def _register_builders() -> None:
    if _BUILDERS:
        return
    from jarl.agentic.langgraph.graphs.subagents import metrics_analysis

    _BUILDERS[metrics_analysis.GRAPH_ID] = metrics_analysis.build_graph


def subgraph_graph_ids() -> frozenset[str]:
    """Return registered subagent graph identifiers."""
    _register_builders()
    return frozenset(_BUILDERS)


def get_subgraph_builder(graph_id: str) -> SubgraphBuilder:
    """Return the builder for ``graph_id``.

    Raises:
        KeyError: If the subgraph is not registered.
    """
    _register_builders()
    try:
        return _BUILDERS[graph_id]
    except KeyError as exc:
        msg = f"Unknown subagent graph id: {graph_id!r}"
        raise KeyError(msg) from exc
