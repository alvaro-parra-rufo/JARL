"""Tests for LangChain tool adapters used by the experiment graph."""

from __future__ import annotations

from pathlib import Path

from jarl.agentic.langgraph.graphs.experiment import OPERATE_INCLUDE_LABELS
from jarl.agentic.langgraph.tools import build_langchain_tools
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow


class TestBoundCatalogFidelity:
    def test_bound_catalog_matches_structured_tools_from_build_langchain_tools(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        registry = REGISTRY.filter_by(include_labels=set(OPERATE_INCLUDE_LABELS))
        structured = {tool.name: tool for tool in build_langchain_tools(registry, workflow)}
        catalog = {item.name: item for item in registry.bound_catalog()}

        assert catalog.keys() == structured.keys()
        for name, bound in catalog.items():
            tool = structured[name]
            assert bound.description == tool.description
            assert bound.schema == tool.tool_call_schema
