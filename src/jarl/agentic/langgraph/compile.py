"""Compile and invoke LangGraph workflows for agentic experiments."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from jarl.agentic.audit import AGENTIC_DIRNAME, RunTracker, SubgraphRunLink, subagent_run_dir
from jarl.agentic.langgraph.graphs.experiment import GRAPH_ID as EXPERIMENT_GRAPH_ID
from jarl.agentic.langgraph.graphs.experiment import build_graph as build_experiment_graph
from jarl.agentic.langgraph.graphs.experiment import resolve_experiment_phase
from jarl.agentic.langgraph.registry import get_subgraph_builder
from jarl.agentic.langgraph.state import AgentState
from jarl.agentic.langgraph.tools import build_langchain_tools
from jarl.agentic.langgraph.trace import LlmGenerationCallback
from jarl.agentic.llm import MAIN_COMPONENT_ID, ResolvedLLM
from jarl.agentic.run_events import (
    ExperimentInvokeEvent,
    ExperimentResultEvent,
    GraphNodeEvent,
    SubgraphInvokeEvent,
    SubgraphResultEvent,
)
from jarl.agentic.tools.specs import ToolFilterBy
from jarl.agentic.transcript.pointer import ActiveCheckpointPointer

if TYPE_CHECKING:
    from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "LANGGRAPH_CHECKPOINT_DB_FILENAME",
    "build_checkpointer",
    "build_langchain_tools",
    "compile_experiment_graph",
    "invoke_experiment",
    "invoke_subgraph",
    "langgraph_checkpoint_db_path",
    "langgraph_checkpoint_dir",
]

LANGGRAPH_CHECKPOINT_DB_FILENAME = "checkpoints.sqlite"
"""SQLite database filename under ``langgraph_checkpoint_dir``."""

_ROUTING_ONLY_NODES: frozenset[str] = frozenset({"bootstrap"})
"""LangGraph nodes that route without binding an agent; they emit no ``graph_node``."""

_checkpointer_cache: dict[Path, SqliteSaver] = {}


def langgraph_checkpoint_dir(exp_dir: Path) -> Path:
    """Return ``<exp_dir>/.agentic/langgraph`` for LangGraph checkpoints."""
    return exp_dir.resolve() / AGENTIC_DIRNAME / "langgraph"


def langgraph_checkpoint_db_path(exp_dir: Path) -> Path:
    """Return the SQLite checkpoint database path for an experiment."""
    return langgraph_checkpoint_dir(exp_dir) / LANGGRAPH_CHECKPOINT_DB_FILENAME


def build_checkpointer(exp_dir: Path) -> SqliteSaver:
    """Build a SQLite-backed checkpointer for ``exp_dir``.

    Reuses an open saver per experiment directory within the current process so
    repeated ``invoke`` calls share the same connection and on-disk state.
    """
    resolved = exp_dir.resolve()
    cached = _checkpointer_cache.get(resolved)
    if cached is not None:
        return cached

    checkpoint_dir = langgraph_checkpoint_dir(resolved)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(langgraph_checkpoint_db_path(resolved)),
        check_same_thread=False,
    )
    saver = SqliteSaver(conn)
    saver.setup()
    _checkpointer_cache[resolved] = saver
    return saver


def compile_experiment_graph(
    workflow: AgenticWorkflow,
    llm: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    filter_by: ToolFilterBy | None = None,
    objective: str | None = None,
) -> CompiledStateGraph:
    """Compile the main experiment LangGraph for ``workflow``."""
    active_checkpointer = checkpointer or build_checkpointer(workflow.exp_dir)
    ctx = workflow.build_tool_context()
    return build_experiment_graph(
        ctx,
        ctx.registry,
        llm,
        checkpointer=active_checkpointer,
        filter_by=filter_by,
        objective=objective,
    )


def invoke_experiment(
    workflow: AgenticWorkflow,
    *,
    input_state: dict[str, object] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    filter_by: ToolFilterBy | None = None,
    objective: str | None = None,
) -> AgentState:
    """Compile and invoke the experiment graph for one agent turn."""
    active_checkpointer = checkpointer or build_checkpointer(workflow.exp_dir)
    llm = workflow.get_chat_model(MAIN_COMPONENT_ID)
    resolved = workflow.last_resolved_llm
    compiled = compile_experiment_graph(
        workflow,
        llm,
        checkpointer=active_checkpointer,
        filter_by=filter_by,
        objective=objective,
    )
    payload = input_state or {}
    raw_messages = payload.get("messages", [])
    state = AgentState(
        messages=raw_messages if isinstance(raw_messages, list) else [],
        experiment_dir=str(workflow.exp_dir),
        thread_id=workflow.thread_id,
    )
    tracker = workflow.open_run(
        run_kind="agent",
        graph_id=EXPERIMENT_GRAPH_ID,
        llm_profile=None if resolved is None else resolved.profile_name,
        llm_provider=None if resolved is None else resolved.settings.provider,
        llm_model=None if resolved is None else resolved.settings.model,
    )
    config = _runnable_config(workflow.thread_id, tracker)
    tracker.append_event(
        ExperimentInvokeEvent(
            message_count=state.message_count,
            phase=resolve_experiment_phase(workflow),
            experiment_node_id=workflow.try_current_node_id(),
            **_llm_event_fields(resolved),
        )
    )
    pointer = ActiveCheckpointPointer.for_thread(workflow.exp_dir, workflow.thread_id)
    with pointer:
        try:
            values, nodes_visited = _stream_and_record_graph_nodes(
                compiled,
                state.to_payload(),
                config=config,
                tracker=tracker,
                pointer=pointer,
                subgraphs=True,
            )
            output = AgentState.from_mapping(values)
            tracker.append_event(
                ExperimentResultEvent(
                    message_count=output.message_count,
                    phase=resolve_experiment_phase(workflow),
                    experiment_node_id=workflow.try_current_node_id(),
                    nodes_visited=list(nodes_visited),
                )
            )
            tracker.complete()
            return output
        except Exception:
            tracker.fail()
            raise


def invoke_subgraph(
    workflow: AgenticWorkflow,
    graph_id: str,
    input_state: dict[str, object],
    *,
    thread_suffix: str | None = None,
    parent_tool: str | None = None,
) -> dict[str, object]:
    """Compile (if needed), invoke a subagent graph, and track the child run."""
    suffix = thread_suffix or graph_id.rsplit("/", maxsplit=1)[-1]
    child_thread = f"{workflow.thread_id}:{suffix}"
    child_root = subagent_run_dir(workflow.exp_dir, workflow.thread_id, suffix)
    workflow.set_last_subgraph_link(
        SubgraphRunLink(
            sub_thread_id=child_thread,
            parent_tool=parent_tool,
            run_dir=child_root.resolve().relative_to(workflow.exp_dir.resolve()).as_posix(),
        )
    )
    llm = workflow.get_chat_model(graph_id)
    resolved = workflow.last_resolved_llm
    tracker = workflow.open_run(
        child_thread,
        parent_thread_id=workflow.thread_id,
        run_kind="subagent",
        graph_id=graph_id,
        parent_tool=parent_tool,
        thread_suffix=suffix,
        llm_profile=None if resolved is None else resolved.profile_name,
        llm_provider=None if resolved is None else resolved.settings.provider,
        llm_model=None if resolved is None else resolved.settings.model,
    )
    tracker.append_event(
        SubgraphInvokeEvent(
            graph_id=graph_id,
            input_keys=sorted(input_state),
            **_llm_event_fields(resolved),
        )
    )
    try:
        ctx = workflow.build_tool_context()
        builder = get_subgraph_builder(graph_id)
        compiled = builder(ctx, ctx.registry, llm)
        config = _runnable_config(child_thread, tracker)
        values, _nodes_visited = _stream_and_record_graph_nodes(
            compiled,
            {"input": input_state},
            config=config,
            tracker=tracker,
        )
        output = values.get("output", values)
        if not isinstance(output, dict):
            msg = f"Subgraph {graph_id!r} returned a non-dict output."
            raise TypeError(msg)
        tracker.append_event(
            SubgraphResultEvent(output_keys=sorted(output)),
        )
        tracker.complete()
        return output
    except Exception:
        tracker.fail()
        raise


def _runnable_config(thread_id: str, tracker: RunTracker) -> RunnableConfig:
    """Return stream config with thread id and always-on LLM metadata callbacks."""
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": [LlmGenerationCallback(tracker)],
    }


def _llm_event_fields(resolved: ResolvedLLM | None) -> dict[str, str]:
    """Return audit event fields for a catalog resolution, if any."""
    if resolved is None:
        return {}
    return {
        "llm_profile": resolved.profile_name,
        "llm_provider": resolved.settings.provider,
        "llm_model": resolved.settings.model,
    }


def _stream_and_record_graph_nodes(
    compiled: CompiledStateGraph,
    payload: dict[str, object],
    *,
    config: RunnableConfig,
    tracker: RunTracker,
    pointer: ActiveCheckpointPointer | None = None,
    subgraphs: bool = False,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Stream graph updates and append ``graph_node`` events for agent nodes.

    When ``subgraphs`` is true, nested ReAct namespaces update ``pointer`` so
    readers can load the child checkpoint mid-turn. Nested ``model``/``tools``
    updates do not emit parent ``graph_node`` rows.
    """
    visited: list[str] = []
    last_values: dict[str, object] = {}
    can_checkpoint = compiled.checkpointer is not None
    for namespace, data in _iter_stream_update_chunks(
        compiled,
        payload,
        config=config,
        subgraphs=subgraphs,
    ):
        if namespace:
            if pointer is not None:
                pointer.set(_format_checkpoint_ns(namespace))
            continue
        for node_name, update in data.items():
            name = str(node_name)
            if name in _ROUTING_ONLY_NODES or update is None:
                continue
            if isinstance(update, Mapping) and not update:
                continue
            last_values = _channel_values_after_update(
                compiled,
                update,
                config=config,
                can_checkpoint=can_checkpoint,
            )
            tracker.append_event(
                GraphNodeEvent(
                    node=name,
                    message_count=AgentState.from_mapping(last_values).message_count,
                )
            )
            visited.append(name)
    if can_checkpoint:
        snapshot = compiled.get_state(config).values
        last_values = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    return last_values, tuple(visited)


def _iter_stream_update_chunks(
    compiled: CompiledStateGraph,
    payload: dict[str, object],
    *,
    config: RunnableConfig,
    subgraphs: bool,
) -> Iterator[tuple[tuple[str, ...], Mapping[str, object]]]:
    """Yield ``(checkpoint_namespace, updates)`` from a compiled graph stream."""
    for chunk in compiled.stream(
        payload,
        config=config,
        stream_mode="updates",
        subgraphs=subgraphs,
    ):
        if subgraphs:
            if not isinstance(chunk, tuple):
                continue
            try:
                raw_namespace, data = chunk
            except ValueError:
                continue
            if not isinstance(data, Mapping):
                continue
            yield _coerce_stream_namespace(raw_namespace), data
            continue
        if isinstance(chunk, Mapping):
            yield (), chunk


def _coerce_stream_namespace(raw_namespace: object) -> tuple[str, ...]:
    """Normalize a LangGraph stream namespace tuple to strings."""
    if not isinstance(raw_namespace, tuple):
        return ()
    return tuple(str(part) for part in raw_namespace)


def _format_checkpoint_ns(namespace: Sequence[str]) -> str:
    """Format a stream namespace as a SqliteSaver ``checkpoint_ns`` string."""
    if not namespace:
        return ""
    if len(namespace) == 1:
        return namespace[0]
    return "|".join(namespace)


def _channel_values_after_update(
    compiled: CompiledStateGraph,
    update: object,
    *,
    config: RunnableConfig,
    can_checkpoint: bool,
) -> dict[str, object]:
    """Return channel values after one streamed node update.

    Parent ``get_state`` can lag a nested ``create_agent`` subgraph: the
    ``updates`` chunk already carries the ReAct messages while the parent
    checkpoint still has the pre-subgraph list. Prefer the longer ``messages``
    channel so ``graph_node.message_count`` matches the conversation.
    """
    if can_checkpoint:
        snapshot = compiled.get_state(config).values
        values = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    elif isinstance(update, Mapping):
        values = dict(update)
    else:
        values = {}
    if not isinstance(update, Mapping):
        return values
    update_messages = update.get("messages")
    snapshot_messages = values.get("messages")
    update_count = len(update_messages) if isinstance(update_messages, list) else 0
    snapshot_count = len(snapshot_messages) if isinstance(snapshot_messages, list) else 0
    if update_count > snapshot_count:
        return {**values, "messages": update_messages}
    return values
