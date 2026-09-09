"""Agentic workflow executor: session, graph lifecycle, and audit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from jarl.agentic.audit import (
    AuditIndexEvent,
    RunKind,
    RunTracker,
    SubgraphRunLink,
    append_audit_index,
    open_run,
)
from jarl.agentic.errors import ExperimentNotFoundError, GraphMismatchError, LangGraphNotConfiguredError
from jarl.agentic.langgraph.state import AgentState
from jarl.agentic.llm import (
    MAIN_COMPONENT_ID,
    LLMCatalog,
    ResolvedLLM,
    catalog_from_settings,
    create_chat_model,
    effective_settings_key,
    load_llm_catalog,
)
from jarl.agentic.llm.config import LLMSettings
from jarl.agentic.session import AgenticSession, CompiledNodeInfo, load_or_create_session, save_session
from jarl.agentic.tools.context import ToolContext
from jarl.agentic.tools.specs import ToolFilterBy
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.manifest import MANIFEST_FILENAME
from jarl.training.config import RLRunConfig

__all__ = ["AgenticWorkflow"]


class AgenticWorkflow:
    """Session-aware executor over an experiment graph.

    Loads ``ExperimentGraph`` in memory, persists agent session metadata, and
    records audit events.
    """

    def __init__(
        self,
        exp_dir: Path,
        *,
        graph: ExperimentGraph[RLRunConfig],
        session: AgenticSession,
    ) -> None:
        """Initialize a workflow for an existing experiment and session."""
        self._exp_dir = exp_dir.resolve()
        self._graph = graph
        self._session = session
        self._last_subgraph_link: SubgraphRunLink | None = None
        self._llm: BaseChatModel | None = None
        self._llm_catalog: LLMCatalog | None = None
        self._llm_clients: dict[tuple[object, ...], BaseChatModel] = {}
        self._last_resolved_llm: ResolvedLLM | None = None
        self._tool_overrides: dict[str, dict[str, object]] = {}

    @classmethod
    def from_experiment(
        cls,
        exp_dir: str | Path,
        *,
        audit_reads: bool | None = None,
        tool_overrides: dict[str, dict[str, object]] | None = None,
    ) -> AgenticWorkflow:
        """Open any experiment directory with a valid ``experiment.json`` manifest.

        Args:
            exp_dir: Experiment root directory.
            audit_reads: When set, enable or disable auditing read tools. Omitted
                values leave an existing session unchanged.
            tool_overrides: Optional per-tool request fields forced during this
                workflow session. LLM-provided values are overwritten.

        Returns:
            A workflow bound to the experiment graph and agent session.

        Raises:
            ExperimentNotFoundError: If the manifest is missing.
        """
        resolved = Path(exp_dir).resolve()
        manifest_path = resolved / MANIFEST_FILENAME
        if not manifest_path.is_file():
            msg = f"Experiment manifest not found: {manifest_path}"
            raise ExperimentNotFoundError(msg)

        graph = ExperimentGraph.from_directory(resolved, config_cls=RLRunConfig)
        session = load_or_create_session(resolved, audit_reads=audit_reads)
        workflow = cls(resolved, graph=graph, session=session)
        if tool_overrides:
            workflow._tool_overrides = {tool: dict(fields) for tool, fields in tool_overrides.items()}
        return workflow

    @property
    def exp_dir(self) -> Path:
        """Experiment root directory."""
        return self._exp_dir

    @property
    def graph(self) -> ExperimentGraph[RLRunConfig]:
        """In-memory experiment graph."""
        return self._graph

    @property
    def session(self) -> AgenticSession:
        """Agent session metadata."""
        return self._session

    def tool_overrides_for(self, tool_name: str) -> dict[str, object]:
        """Return forced request fields for ``tool_name`` during this workflow."""
        return dict(self._tool_overrides.get(tool_name, {}))

    @property
    def thread_id(self) -> str:
        """LangGraph thread id for this session."""
        return self._session.thread_id

    @property
    def current_node_id(self) -> str:
        """Current node id from the experiment manifest."""
        return self.reload_graph().current_node.id

    def try_current_node_id(self) -> str | None:
        """Return the current node id when set, otherwise ``None``."""
        graph = self.reload_graph()
        try:
            return graph.current_node.id
        except RuntimeError:
            return None

    def reload_graph(self) -> ExperimentGraph[RLRunConfig]:
        """Reload ``ExperimentGraph`` from disk.

        The experiment manifest is the source of truth for the current node.
        """
        self._graph = ExperimentGraph.from_directory(self._exp_dir, config_cls=RLRunConfig)
        return self._graph

    def replace_graph(self, graph: ExperimentGraph[RLRunConfig]) -> None:
        """Replace the in-memory graph after a mutating operation or train step.

        Args:
            graph: Updated graph for the same experiment directory.

        Raises:
            GraphMismatchError: If ``graph`` belongs to a different experiment.
        """
        if graph.layout.root.resolve() != self._exp_dir:
            msg = f"Graph directory {graph.layout.root!r} does not match workflow experiment {self._exp_dir!r}."
            raise GraphMismatchError(msg)
        self._graph = graph

    def touch_session(self) -> None:
        """Persist an updated ``updated_at`` timestamp for the agent session."""
        self._session = self._session.touch()
        save_session(self._exp_dir, self._session)

    def replace_compiled_nodes(self, nodes: Mapping[str, CompiledNodeInfo]) -> None:
        """Replace the session compile snapshot with ``nodes``.

        Skips a write when the snapshot already matches, so repeated compiles
        do not bump ``updated_at``.
        """
        incoming = dict(nodes)
        if self._session.compiled_nodes == incoming:
            return
        self._session = self._session.with_compiled_nodes(incoming)
        save_session(self._exp_dir, self._session)

    def record_audit(self, event: AuditIndexEvent) -> None:
        """Append a summarized event to ``agentic_audit.jsonl``."""
        append_audit_index(self._exp_dir, event)

    def open_run(
        self,
        thread_id: str | None = None,
        *,
        parent_thread_id: str | None = None,
        run_kind: RunKind = "agent",
        graph_id: str = "experiment",
        parent_tool: str | None = None,
        thread_suffix: str | None = None,
        llm_profile: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> RunTracker:
        """Open detailed tracking for an agent or nested subagent run."""
        active_thread = thread_id or self.thread_id
        return open_run(
            self._exp_dir,
            active_thread,
            parent_thread_id=parent_thread_id,
            run_kind=run_kind,
            graph_id=graph_id,
            parent_tool=parent_tool,
            thread_suffix=thread_suffix,
            llm_profile=llm_profile,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    @property
    def llm(self) -> BaseChatModel | None:
        """LLM override bound for the current agent turn, if any."""
        return self._llm

    @property
    def llm_catalog(self) -> LLMCatalog | None:
        """Bound LLM catalog, when configured."""
        return self._llm_catalog

    @property
    def last_resolved_llm(self) -> ResolvedLLM | None:
        """Last catalog resolution produced by ``get_chat_model``."""
        return self._last_resolved_llm

    def set_llm(self, llm: BaseChatModel | None) -> None:
        """Bind or clear a global LLM override for the current turn.

        When set, every ``get_chat_model`` call returns this instance and skips
        catalog resolution. Used by tests and ``invoke(llm=...)``.
        """
        self._llm = llm
        if llm is not None:
            self._last_resolved_llm = None

    def set_llm_catalog(self, catalog: LLMCatalog | None) -> None:
        """Bind or clear the LLM catalog used for component resolution."""
        self._llm_catalog = catalog
        self._llm_clients.clear()
        self._last_resolved_llm = None

    def get_chat_model(self, component_id: str = MAIN_COMPONENT_ID) -> BaseChatModel:
        """Return the chat model for ``component_id``.

        Prefer the turn override from ``set_llm`` / ``invoke(llm=...)`` when
        present. Otherwise resolve ``component_id`` against the bound catalog
        (or a lazily loaded default catalog) and cache clients by effective
        settings identity.
        """
        if self._llm is not None:
            self._last_resolved_llm = None
            return self._llm

        catalog = self._llm_catalog
        if catalog is None:
            catalog = load_llm_catalog()
            self._llm_catalog = catalog

        resolved = catalog.resolve(component_id)
        self._last_resolved_llm = resolved
        cache_key = effective_settings_key(resolved.settings)
        cached = self._llm_clients.get(cache_key)
        if cached is not None:
            return cached
        client = create_chat_model(resolved.settings)
        self._llm_clients[cache_key] = client
        return client

    def set_last_subgraph_link(self, link: SubgraphRunLink) -> None:
        """Remember the latest subagent run link for index enrichment."""
        self._last_subgraph_link = link

    def take_last_subgraph_link(self) -> SubgraphRunLink | None:
        """Return and clear the latest subagent run link, if any."""
        link = self._last_subgraph_link
        self._last_subgraph_link = None
        return link

    def build_tool_context(self) -> ToolContext:
        """Build a ``ToolContext`` for tool handlers."""
        return ToolContext.from_workflow(self)

    def invoke_subgraph(
        self,
        graph_id: str,
        input_state: dict[str, object],
        *,
        thread_suffix: str | None = None,
        parent_tool: str | None = None,
    ) -> dict[str, object]:
        """Invoke a registered subagent LangGraph and return its compact output."""
        from jarl.agentic.langgraph.compile import invoke_subgraph as _invoke_subgraph

        return _invoke_subgraph(
            self,
            graph_id,
            input_state,
            thread_suffix=thread_suffix,
            parent_tool=parent_tool,
        )

    def invoke(
        self,
        input_state: dict[str, object] | None = None,
        *,
        llm: BaseChatModel | None = None,
        llm_settings: LLMSettings | None = None,
        llm_catalog: LLMCatalog | None = None,
        filter_by: ToolFilterBy | None = None,
        objective: str | None = None,
    ) -> AgentState:
        """Compile and invoke the experiment LangGraph workflow.

        Pass ``llm`` to force a global override for the turn (tests). Otherwise
        the workflow resolves models from ``llm_catalog``, a previously bound
        catalog, ``llm_settings`` (single-profile synthesis), or the default
        YAML/env catalog loader.

        ``filter_by`` is applied after the phase tool filter.
        ``objective`` is included in both phase system prompts when set.
        """
        if llm is not None and (llm_settings is not None or llm_catalog is not None):
            msg = "Pass either llm or catalog/settings, not both."
            raise ValueError(msg)
        if llm_settings is not None and llm_catalog is not None:
            msg = "Pass either llm_settings or llm_catalog, not both."
            raise ValueError(msg)

        previous_override = self._llm
        previous_catalog = self._llm_catalog
        if llm is not None:
            self._llm = llm
        elif llm_catalog is not None:
            self.set_llm_catalog(llm_catalog)
        elif llm_settings is not None:
            self.set_llm_catalog(catalog_from_settings(llm_settings))
        elif self._llm is None and self._llm_catalog is None:
            try:
                self.set_llm_catalog(load_llm_catalog())
            except Exception as exc:
                msg = "An LLM instance or LLM catalog is required for workflow.invoke"
                raise LangGraphNotConfiguredError(msg) from exc

        if self._llm is None and self._llm_catalog is None:
            msg = "An LLM instance or LLM catalog is required for workflow.invoke"
            raise LangGraphNotConfiguredError(msg)

        from jarl.agentic.langgraph.compile import invoke_experiment

        try:
            return invoke_experiment(
                self,
                input_state=input_state,
                filter_by=filter_by,
                objective=objective,
            )
        finally:
            self._llm = previous_override
            if llm is not None or llm_settings is not None or llm_catalog is not None:
                self._llm_catalog = previous_catalog
                if llm is not None:
                    self._llm_clients.clear()
                    self._last_resolved_llm = None
