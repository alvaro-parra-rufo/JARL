"""Registry completeness tests for the agentic tools catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow

V1_TOOL_CATALOG: dict[str, tuple[frozenset[str], str]] = {
    "graph_create_root": (
        frozenset({"graph", "initialization"}),
        "jarl.agentic.tools.graph.create_root",
    ),
    "graph_summary": (
        frozenset({"graph", "read"}),
        "jarl.agentic.tools.graph.summary",
    ),
    "graph_checkpoints": (
        frozenset({"graph", "read"}),
        "jarl.agentic.tools.graph.checkpoints",
    ),
    "graph_checkpoint_rollout": (
        frozenset({"graph", "read", "inference"}),
        "jarl.agentic.tools.graph.checkpoint_rollout",
    ),
    "graph_subtree": (
        frozenset({"graph", "read"}),
        "jarl.agentic.tools.graph.subtree",
    ),
    "graph_diff": (
        frozenset({"graph", "read"}),
        "jarl.agentic.tools.graph.diff",
    ),
    "graph_reward": (
        frozenset({"graph", "read"}),
        "jarl.agentic.tools.graph.reward",
    ),
    "graph_set_reward": (
        frozenset({"graph", "mutation"}),
        "jarl.agentic.tools.graph.set_reward",
    ),
    "graph_checkout": (
        frozenset({"graph", "mutation"}),
        "jarl.agentic.tools.graph.checkout",
    ),
    "graph_fork": (
        frozenset({"graph", "mutation"}),
        "jarl.agentic.tools.graph.fork",
    ),
    "graph_extend": (
        frozenset({"graph", "mutation"}),
        "jarl.agentic.tools.graph.extend",
    ),
    "train_run": (
        frozenset({"train"}),
        "jarl.agentic.tools.train.run",
    ),
    "train_resume": (
        frozenset({"train"}),
        "jarl.agentic.tools.train.resume",
    ),
    "train_recovery_status": (
        frozenset({"train", "read"}),
        "jarl.agentic.tools.train.recovery_status",
    ),
    "session_status": (
        frozenset({"session", "read"}),
        "jarl.agentic.tools.session.status",
    ),
    "session_finish": (
        frozenset({"session", "finish"}),
        "jarl.agentic.tools.session.finish",
    ),
    "subagent_metrics_analysis": (
        frozenset({"subagent", "graph", "read"}),
        "jarl.agentic.tools.subagent.metrics_analysis",
    ),
    "env_navix_maps": (
        frozenset({"env", "read"}),
        "jarl.agentic.tools.env.navix_maps",
    ),
}


class TestRegistryV1Completeness:
    def test_registry_matches_v1_catalog_names(self) -> None:
        assert REGISTRY.names() == frozenset(V1_TOOL_CATALOG)

    @pytest.mark.parametrize(
        ("tool_name", "required_labels", "module_path"),
        [(name, labels, module) for name, (labels, module) in sorted(V1_TOOL_CATALOG.items())],
    )
    def test_v1_tool_metadata_matches_catalog(
        self,
        tool_name: str,
        required_labels: frozenset[str],
        module_path: str,
    ) -> None:
        entry = REGISTRY.get(tool_name)
        assert entry.spec.name == tool_name
        assert required_labels <= entry.spec.labels
        assert entry.spec.module == module_path

    @pytest.mark.parametrize("tool_name", sorted(V1_TOOL_CATALOG))
    def test_v1_tool_builds_handler(self, tool_name: str, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        handler = REGISTRY.get(tool_name).build_handler(workflow)
        assert callable(handler)
