"""Build a model-debug timeline from experiment run artefacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from jarl.agentic.audit import load_run_events
from jarl.agentic.langgraph.graphs.experiment import resolve_experiment_phase
from jarl.agentic.run_events import LlmEndEvent, LlmErrorEvent, RunWindows
from jarl.agentic.session import AgenticSession, CompiledNodeInfo, load_session
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.tools.specs import BoundTool
from jarl.agentic.transcript import ConversationTranscript, TranscriptMessage
from jarl.agentic.workflow import AgenticWorkflow

__all__ = [
    "DebugContext",
    "DebugSegment",
    "ModelDebugView",
    "attach_segment_context",
    "segment_transcript",
]


@dataclass(frozen=True, slots=True)
class DebugContext:
    """LangGraph bind snapshot attached to one transcript segment."""

    node: str
    experiment_node_id: str | None
    system_prompt: str | None
    tools: tuple[BoundTool, ...]
    inferred: bool
    generations: tuple[LlmEndEvent, ...] = ()
    errors: tuple[LlmErrorEvent, ...] = ()

    @classmethod
    def placeholder(
        cls,
        node: str,
        *,
        experiment_node_id: str | None,
        inferred: bool,
    ) -> DebugContext:
        """Return a context shell filled later by ``attach_segment_context``."""
        return cls(
            node=node,
            experiment_node_id=experiment_node_id,
            system_prompt=None,
            tools=(),
            inferred=inferred,
        )


@dataclass(frozen=True, slots=True)
class DebugSegment:
    """A contiguous transcript slice with the bind context of that tramo."""

    context: DebugContext
    messages: tuple[TranscriptMessage, ...]


@dataclass(frozen=True, slots=True)
class ModelDebugView:
    """Sticky timeline of model-facing context for an experiment session."""

    phase: str | None
    experiment_node_id: str | None
    running: bool
    segments: tuple[DebugSegment, ...]

    @classmethod
    def from_workflow(cls, workflow: AgenticWorkflow, *, running: bool = False) -> ModelDebugView:
        """Assemble the model-debug timeline for an open ``AgenticWorkflow``."""
        session = workflow.session
        windows = RunWindows.from_events(load_run_events(workflow.exp_dir, workflow.thread_id))
        transcript = ConversationTranscript.load(workflow.exp_dir)
        raw_segments = segment_transcript(transcript, windows, running=running)
        fallback_node = _sticky_phase(windows, workflow, session)
        segments = attach_segment_context(
            raw_segments,
            session.compiled_nodes,
            fallback_node=fallback_node,
            windows=windows,
            running=running,
        )
        last_result = windows.last_result
        experiment_node_id = (
            last_result.experiment_node_id if last_result is not None else workflow.try_current_node_id()
        )
        return cls(
            phase=fallback_node,
            experiment_node_id=experiment_node_id,
            running=running,
            segments=segments,
        )

    @classmethod
    def from_experiment(cls, exp_dir: Path, *, running: bool = False) -> ModelDebugView:
        """Assemble the timeline for ``exp_dir`` without creating a session.

        Missing ``agentic_session.json`` yields an empty view. Callers that already
        hold an ``AgenticWorkflow`` should use ``from_workflow``.
        """
        resolved = exp_dir.resolve()
        if load_session(resolved) is None:
            return cls(phase=None, experiment_node_id=None, running=running, segments=())
        return cls.from_workflow(AgenticWorkflow.from_experiment(resolved), running=running)


def segment_transcript(
    transcript: ConversationTranscript | None,
    windows: RunWindows,
    *,
    running: bool = False,
) -> tuple[DebugSegment, ...]:
    """Split transcript messages into segments keyed by LangGraph node.

    When ``running`` is true, checkpoint ``message_count`` cuts are ignored and
    the live transcript is one in-flight segment (last ``llm_start`` or
    ``graph_node``).
    """
    messages = () if transcript is None else transcript.messages
    graph_nodes = windows.graph_nodes
    if running:
        last_window = windows.windows[-1] if windows.windows else None
        last_start = last_window.llm_starts[-1] if last_window and last_window.llm_starts else None
        last_graph = last_window.graph_nodes[-1] if last_window and last_window.graph_nodes else None
        if last_start is not None and last_start.node:
            node_name = last_start.node
            inferred = False
        elif last_graph is not None:
            node_name = last_graph.node
            inferred = False
        else:
            node_name = ""
            inferred = True
        return (
            DebugSegment(
                context=DebugContext.placeholder(
                    node_name,
                    experiment_node_id=windows.sticky_experiment_node_id,
                    inferred=inferred,
                ),
                messages=messages,
            ),
        )

    if not graph_nodes:
        return (
            DebugSegment(
                context=DebugContext.placeholder(
                    "",
                    experiment_node_id=windows.sticky_experiment_node_id,
                    inferred=True,
                ),
                messages=messages,
            ),
        )

    segments: list[DebugSegment] = []
    start = 0
    for window in windows:
        for graph_node in window.graph_nodes:
            end = min(graph_node.message_count, len(messages))
            segments.append(
                DebugSegment(
                    context=DebugContext.placeholder(
                        graph_node.node,
                        experiment_node_id=window.experiment_node_id,
                        inferred=False,
                    ),
                    messages=messages[start:end],
                )
            )
            start = end
    return tuple(segments)


def attach_segment_context(
    segments: Sequence[DebugSegment],
    compiled_nodes: Mapping[str, CompiledNodeInfo],
    *,
    fallback_node: str | None = None,
    windows: RunWindows | None = None,
    running: bool = False,
) -> tuple[DebugSegment, ...]:
    """Fill system prompt, bound tools, and LLM generations from artefacts.

    Uses each segment's LangGraph node name. When that name is empty, ``fallback_node``
    is used so inferred runs still attach the last compile bind.
    """
    attached: list[DebugSegment] = []
    llm_by_index = _llm_events_by_segment(segments, windows, running=running)
    for index, segment in enumerate(segments):
        node_name = segment.context.node or (fallback_node or "")
        info = compiled_nodes.get(node_name)
        tools = () if info is None else REGISTRY.bound_catalog(**info.as_filter_by())
        generations, errors = llm_by_index[index]
        context = replace(
            segment.context,
            node=node_name,
            system_prompt=None if info is None else info.system_prompt,
            tools=tools,
            generations=generations,
            errors=errors,
        )
        attached.append(replace(segment, context=context))
    return tuple(attached)


def _llm_events_by_segment(
    segments: Sequence[DebugSegment],
    windows: RunWindows | None,
    *,
    running: bool,
) -> list[tuple[tuple[LlmEndEvent, ...], tuple[LlmErrorEvent, ...]]]:
    """Align ``llm_end`` / ``llm_error`` rows to transcript segments."""
    empty: tuple[tuple[LlmEndEvent, ...], tuple[LlmErrorEvent, ...]] = ((), ())
    if windows is None or not segments:
        return [empty] * len(segments)
    if running or (len(segments) == 1 and segments[0].context.inferred):
        last = windows.windows[-1] if windows.windows else None
        if last is None:
            return [empty]
        node_name = segments[0].context.node
        ends = tuple(event for event in last.llm_ends if not node_name or event.node == node_name)
        errors = tuple(event for event in last.llm_errors if not node_name or event.node == node_name)
        return [(ends, errors)]
    pairs = [(window, node) for window in windows for node in window.graph_nodes]
    assigned: list[tuple[tuple[LlmEndEvent, ...], tuple[LlmErrorEvent, ...]]] = []
    for index in range(len(segments)):
        if index >= len(pairs):
            assigned.append(empty)
            continue
        window, graph_node = pairs[index]
        assigned.append(
            (
                tuple(event for event in window.llm_ends if event.node == graph_node.node),
                tuple(event for event in window.llm_errors if event.node == graph_node.node),
            )
        )
    return assigned


def _sticky_phase(
    windows: RunWindows,
    workflow: AgenticWorkflow,
    session: AgenticSession,
) -> str | None:
    """Return the LangGraph node name that should label the current view."""
    phase = windows.sticky_phase
    if phase is not None:
        return phase
    resolved_phase = resolve_experiment_phase(workflow)
    if resolved_phase in session.compiled_nodes:
        return resolved_phase
    if session.compiled_nodes:
        return next(reversed(session.compiled_nodes))
    return None
