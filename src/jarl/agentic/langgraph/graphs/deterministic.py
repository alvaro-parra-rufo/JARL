"""Shared LangGraph builders for deterministic subagent modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

__all__ = ["DeterministicSubagentState", "build_deterministic_subgraph"]


class DeterministicSubagentState(TypedDict):
    """Input/output state for deterministic subagent graphs."""

    input: dict[str, object]
    output: dict[str, object]


def build_deterministic_subgraph(
    transform: Callable[[dict[str, object]], BaseModel],
    *,
    node_name: str = "summarize",
) -> CompiledStateGraph:
    """Build a single-node compiled graph that runs ``transform`` without an LLM."""

    def summarize_node(state: DeterministicSubagentState) -> DeterministicSubagentState:
        output = transform(state["input"])
        return {"input": state["input"], "output": output.model_dump(exclude_none=True)}

    graph = StateGraph(DeterministicSubagentState)
    graph.add_node(node_name, summarize_node)
    graph.add_edge(START, node_name)
    graph.add_edge(node_name, END)
    return graph.compile()
