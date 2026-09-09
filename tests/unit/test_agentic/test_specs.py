"""Tests for agentic tool registry and specs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import PropertyMock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from jarl.agentic.audit import audit_index_path
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.handler import wire_handler
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.specs import ToolRegistry, ToolSpec, merge_tool_filters
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig


class TestToolRegistryDiscovery:
    def test_from_module_discovers_helper_tools(self) -> None:
        registry = ToolRegistry.from_module("tests.helpers.agentic_tools")

        assert registry.names() == frozenset({"sample_ping", "sample_mutate"})
        ping = registry.get("sample_ping")
        assert ping.spec.module.endswith("tests.helpers.agentic_tools.ping")
        assert ping.spec.labels == frozenset({"graph", "read"})

    def test_get_missing_tool_raises(self) -> None:
        registry = ToolRegistry()

        with pytest.raises(KeyError):
            registry.get("missing")

    def test_from_module_raises_on_incomplete_leaf_module(self) -> None:
        with pytest.raises(ValueError, match="missing TOOL_SPEC"):
            ToolRegistry.from_module("tests.helpers.agentic_tools_broken")


class TestMergeToolFilters:
    def test_skips_none_and_empty(self) -> None:
        assert merge_tool_filters(None, {}, {"include_labels": set()}) == {}

    def test_unions_non_empty_keys(self) -> None:
        merged = merge_tool_filters(
            {"include_labels": {"setup"}},
            {"exclude_labels": {"finish"}, "include_names": {"graph_fork"}},
        )

        assert merged == {
            "include_labels": {"setup"},
            "exclude_labels": {"finish"},
            "include_names": {"graph_fork"},
        }

    def test_unions_same_key_across_filters(self) -> None:
        merged = merge_tool_filters(
            {"include_labels": {"setup"}},
            {"include_labels": {"session"}},
        )

        assert merged["include_labels"] == {"setup", "session"}


class TestToolRegistryFilter:
    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        return ToolRegistry.from_module("tests.helpers.agentic_tools")

    def test_filter_by_labels(self, registry: ToolRegistry) -> None:
        reads = registry.filter_by(include_labels={"read"})

        assert reads.names() == frozenset({"sample_ping"})

    def test_filter_by_name_exclude(self, registry: ToolRegistry) -> None:
        filtered = registry.filter_by(exclude_names={"sample_mutate"})

        assert filtered.names() == frozenset({"sample_ping"})

    def test_filter_by_labels_and_names(self, registry: ToolRegistry) -> None:
        filtered = registry.filter_by(include_labels={"mutation"}, include_names={"sample_mutate"})

        assert filtered.names() == frozenset({"sample_mutate"})


class TestBoundCatalog:
    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        return ToolRegistry.from_module("tests.helpers.agentic_tools")

    def test_bound_catalog_applies_the_same_filters_as_filter_by(self, registry: ToolRegistry) -> None:
        catalog = registry.bound_catalog(include_labels={"read"})

        assert {item.name for item in catalog} == {"sample_ping"}

    def test_graph_fork_branch_field_description_is_in_the_dumped_schema(self) -> None:
        catalog = REGISTRY.bound_catalog(include_names={"graph_fork"})

        assert len(catalog) == 1
        branch_schema = catalog[0].schema["properties"]["branch"]
        assert isinstance(branch_schema, dict)
        assert branch_schema["description"] == "Name for the new branch."


class TestWireHandler:
    def test_wire_handler_validates_request_and_audits_on_read(
        self,
        tmp_path: Path,
    ) -> None:
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()

        workflow = AgenticWorkflow.from_experiment(exp_dir, audit_reads=True)
        registry = ToolRegistry.from_module("tests.helpers.agentic_tools")
        entry = registry.get("sample_ping")
        handler = entry.build_handler(workflow)

        payload = handler({"text": "hello"})

        assert json.loads(payload)["text"] == "hello"
        lines = audit_index_path(exp_dir).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["tool"] == "sample_ping"
        assert event["kind"] == "read"

    def test_wire_handler_skips_read_audit_when_disabled(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()

        workflow = AgenticWorkflow.from_experiment(exp_dir, audit_reads=False)
        entry = ToolRegistry.from_module("tests.helpers.agentic_tools").get("sample_ping")
        handler = entry.build_handler(workflow)

        handler({"text": "hello"})

        assert not audit_index_path(exp_dir).exists()

    def test_wire_handler_records_error_audit(self, tmp_path: Path) -> None:
        class BadRequest(BaseModel):
            required: str

        spec = ToolSpec(name="bad_tool", description="bad", labels=frozenset({"graph", "read"}))

        def run_bad(ctx: ToolContext, request: BadRequest) -> str:
            del ctx, request
            return "{}"

        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()
        workflow = AgenticWorkflow.from_experiment(exp_dir, audit_reads=True)
        handler = wire_handler(workflow, run_bad, spec, request_cls=BadRequest)

        result = json.loads(handler({}))

        assert result["code"] == "validation"
        assert "required" in result["error"].lower() or "field" in result["error"].lower()
        event = json.loads(audit_index_path(exp_dir).read_text(encoding="utf-8").splitlines()[0])
        assert event["kind"] == "error"
        assert event["result"]["code"] == "validation"

    def test_wire_handler_runs_initialization_tool_on_empty_experiment(
        self,
        empty_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment, audit_reads=False)
        entry = REGISTRY.get("graph_create_root")
        handler = entry.build_handler(workflow)

        payload = handler({"label": "root"})

        result = json.loads(payload)
        assert result["status"] == "prepared"
        assert result["branch"] == "main"
        assert workflow.try_current_node_id() == result["node_id"]

    def test_wire_handler_does_not_access_current_node_id_property(
        self,
        empty_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment, audit_reads=False)
        entry = REGISTRY.get("graph_create_root")
        handler = entry.build_handler(workflow)
        current_node_property = mocker.patch.object(
            AgenticWorkflow,
            "current_node_id",
            new_callable=PropertyMock,
            side_effect=RuntimeError("current_node_id accessed prematurely"),
        )

        handler({"label": "root"})

        current_node_property.assert_not_called()

    def test_initialization_tool_audits_as_mutation_when_audit_reads_disabled(
        self,
        empty_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment, audit_reads=False)
        entry = REGISTRY.get("graph_create_root")
        handler = entry.build_handler(workflow)

        handler({"label": "root"})

        event = json.loads(audit_index_path(empty_experiment).read_text(encoding="utf-8").splitlines()[0])
        assert event["tool"] == "graph_create_root"
        assert event["kind"] == "mutation"
        assert "node_before" not in event
        assert event["node_after"] is not None


class TestBuildHandlers:
    def test_build_handlers_returns_callables(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
        graph.create_root(RLRunConfig(), label="root", prepare=True)
        graph.save()

        workflow = AgenticWorkflow.from_experiment(exp_dir)
        registry = ToolRegistry.from_module("tests.helpers.agentic_tools")
        handlers = registry.build_handlers(workflow)

        assert len(handlers) == 2
        assert all(callable(handler) for handler in handlers)
