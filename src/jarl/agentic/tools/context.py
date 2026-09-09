"""Tool execution context injected by the agentic runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jarl.agentic.audit import AuditIndexEvent, RunKind, RunTracker
from jarl.agentic.session import AgenticSession
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig

if TYPE_CHECKING:
    from jarl.agentic.tools.specs import ToolRegistry
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = ["ToolContext"]


@dataclass(slots=True)
class ToolContext:
    """Context passed to ``run_*`` handlers."""

    workflow: AgenticWorkflow

    @classmethod
    def from_workflow(cls, workflow: AgenticWorkflow) -> ToolContext:
        """Build a context from a workflow."""
        return cls(workflow=workflow)

    @property
    def registry(self) -> ToolRegistry:
        """Project tool registry (``REGISTRY``)."""
        # Deferred import: avoids registry → tools → handler → context → registry cycles.
        from jarl.agentic.tools.registry import REGISTRY

        return REGISTRY

    @property
    def graph(self) -> ExperimentGraph[RLRunConfig]:
        """In-memory experiment graph."""
        return self.workflow.graph

    @property
    def exp_dir(self) -> Path:
        """Experiment root directory."""
        return self.workflow.exp_dir

    @property
    def thread_id(self) -> str:
        """Active LangGraph thread id."""
        return self.workflow.thread_id

    @property
    def current_node_id(self) -> str:
        """Current node id from the experiment manifest."""
        return self.workflow.current_node_id

    def try_current_node_id(self) -> str | None:
        """Return the current node id when set, otherwise ``None``."""
        return self.workflow.try_current_node_id()

    @property
    def session(self) -> AgenticSession:
        """Agent session metadata."""
        return self.workflow.session

    def replace_graph(self, graph: ExperimentGraph[RLRunConfig]) -> None:
        """Replace the workflow graph after a mutation or train step."""
        self.workflow.replace_graph(graph)

    def record_audit(self, event: AuditIndexEvent) -> None:
        """Append a summarized audit index row."""
        self.workflow.record_audit(event)

    def open_run(
        self,
        thread_id: str | None = None,
        *,
        parent_thread_id: str | None = None,
        run_kind: RunKind = "agent",
        graph_id: str = "experiment",
        parent_tool: str | None = None,
        thread_suffix: str | None = None,
    ) -> RunTracker:
        """Open per-run tracking for an agent or nested subagent run."""
        return self.workflow.open_run(
            thread_id,
            parent_thread_id=parent_thread_id,
            run_kind=run_kind,
            graph_id=graph_id,
            parent_tool=parent_tool,
            thread_suffix=thread_suffix,
        )

    def invoke_subgraph(
        self,
        graph_id: str,
        input_state: dict[str, object],
        *,
        thread_suffix: str | None = None,
        parent_tool: str | None = None,
    ) -> dict[str, object]:
        """Invoke a subagent LangGraph."""
        return self.workflow.invoke_subgraph(
            graph_id,
            input_state,
            thread_suffix=thread_suffix,
            parent_tool=parent_tool,
        )
