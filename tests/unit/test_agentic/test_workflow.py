"""Tests for ``AgenticWorkflow`` session and graph lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.agentic.errors import ExperimentNotFoundError, GraphMismatchError, LangGraphNotConfiguredError
from jarl.agentic.session import AGENTIC_SESSION_FILENAME, load_session
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.checkout import CheckoutRequest, checkout
from jarl.training.config import RLRunConfig


class TestAgenticWorkflowBootstrap:
    def test_from_experiment_loads_graph_and_session(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        assert workflow.graph.head("main").id == workflow.current_node_id
        assert workflow.thread_id.startswith("thread_")
        assert load_session(prepared_experiment) is not None

    def test_from_experiment_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExperimentNotFoundError, match="manifest not found"):
            AgenticWorkflow.from_experiment(tmp_path / "missing")


class TestAgenticWorkflowGraphLifecycle:
    def test_reload_graph_reads_current_node_from_manifest(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        root_id = workflow.current_node_id

        graph = ExperimentGraph.from_directory(prepared_experiment, config_cls=RLRunConfig)
        child = graph.fork("alt", label="child", prepare=True)
        graph.save()

        assert workflow.graph.current_node.id == root_id

        workflow.reload_graph()

        assert workflow.current_node_id == child.id

    def test_replace_graph_updates_in_memory_graph(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        graph = workflow.graph
        checkout(graph, CheckoutRequest(node_id=graph.current_node.id))
        child = graph.fork("alt", label="child", prepare=True)
        graph.save()

        workflow.replace_graph(graph)

        assert workflow.graph.current_node.id == child.id

    def test_replace_graph_rejects_foreign_directory(self, prepared_experiment: Path, tmp_path: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        other = ExperimentGraph(tmp_path / "other", base_config=RLRunConfig())
        other.create_root(RLRunConfig(), label="other", prepare=True)

        with pytest.raises(GraphMismatchError, match="does not match"):
            workflow.replace_graph(other)


class TestAgenticWorkflowTooling:
    def test_build_tool_context_delegates_to_workflow(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        ctx = workflow.build_tool_context()

        assert ctx.workflow is workflow
        assert ctx.graph is workflow.graph
        assert ctx.thread_id == workflow.thread_id
        assert "graph_summary" in ctx.registry.names()

    def test_invoke_requires_llm(self, prepared_experiment: Path, mocker: MockerFixture) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        mocker.patch(
            "jarl.agentic.workflow.load_llm_catalog",
            side_effect=RuntimeError("catalog unavailable"),
        )

        with pytest.raises(LangGraphNotConfiguredError, match="LLM instance or LLM catalog"):
            workflow.invoke()


class TestAgenticWorkflowSession:
    def test_touch_session_updates_updated_at(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch(
            "jarl.agentic.session.now_iso",
            side_effect=["2026-07-23T10:00:00+00:00", "2026-07-23T11:00:00+00:00"],
        )

        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        assert workflow.session.created_at == "2026-07-23T10:00:00+00:00"
        assert workflow.session.updated_at == "2026-07-23T10:00:00+00:00"

        workflow.touch_session()

        assert workflow.session.updated_at == "2026-07-23T11:00:00+00:00"
        persisted = json.loads((prepared_experiment / AGENTIC_SESSION_FILENAME).read_text(encoding="utf-8"))
        assert persisted["updated_at"] == "2026-07-23T11:00:00+00:00"
        assert persisted["created_at"] == "2026-07-23T10:00:00+00:00"
