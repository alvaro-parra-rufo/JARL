"""Typed rows persisted to a run ``events.jsonl``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from jarl.metadata import now_iso

__all__ = [
    "ExperimentInvokeEvent",
    "ExperimentResultEvent",
    "ExperimentStamp",
    "GraphNodeEvent",
    "LangGraphNodeStamp",
    "LlmEndEvent",
    "LlmErrorEvent",
    "LlmStartEvent",
    "LlmUsage",
    "RunEvent",
    "RunEventBase",
    "RunWindow",
    "RunWindows",
    "SubgraphInvokeEvent",
    "SubgraphResultEvent",
    "of_kind",
    "parse_run_event",
]


class RunEventBase(BaseModel):
    """Shared envelope for one run-log line."""

    model_config = ConfigDict(frozen=True)

    ts: str = Field(
        default_factory=now_iso,
        description="ISO-8601 timestamp for this event.",
    )
    event: str = Field(description="Event kind discriminator.")


class ExperimentStamp(RunEventBase):
    """Shared location fields for experiment invoke and result events."""

    phase: str = Field(description="Experiment LangGraph phase at this stamp.")
    experiment_node_id: str | None = Field(
        default=None,
        description="Active experiment node id, or ``None`` when the graph is empty.",
    )
    message_count: int = Field(description="Conversation message count at this stamp.")


class LangGraphNodeStamp(RunEventBase):
    """Shared LangGraph node name for graph and LLM events."""

    node: str = Field(description="LangGraph node name that produced this event.")


class ExperimentInvokeEvent(ExperimentStamp):
    """Start of one ``invoke_experiment`` turn."""

    event: Literal["experiment_invoke"] = "experiment_invoke"
    llm_profile: str | None = Field(default=None, description="Resolved LLM profile name.")
    llm_provider: str | None = Field(default=None, description="Resolved LLM provider identifier.")
    llm_model: str | None = Field(default=None, description="Resolved LLM model name.")


class ExperimentResultEvent(ExperimentStamp):
    """End of one ``invoke_experiment`` turn."""

    event: Literal["experiment_result"] = "experiment_result"
    nodes_visited: list[str] = Field(
        description="LangGraph node names that emitted ``graph_node`` in this invoke, in order.",
    )


class GraphNodeEvent(LangGraphNodeStamp):
    """One completed LangGraph agent node during a stream."""

    event: Literal["graph_node"] = "graph_node"
    message_count: int = Field(description="Conversation message count after this node.")


class SubgraphInvokeEvent(RunEventBase):
    """Start of one ``invoke_subgraph`` call."""

    event: Literal["subgraph_invoke"] = "subgraph_invoke"
    graph_id: str = Field(description="Registered subgraph identifier.")
    input_keys: list[str] = Field(description="Sorted keys of the subgraph input mapping.")
    llm_profile: str | None = Field(default=None, description="Resolved LLM profile name.")
    llm_provider: str | None = Field(default=None, description="Resolved LLM provider identifier.")
    llm_model: str | None = Field(default=None, description="Resolved LLM model name.")


class SubgraphResultEvent(RunEventBase):
    """End of one ``invoke_subgraph`` call."""

    event: Literal["subgraph_result"] = "subgraph_result"
    output_keys: list[str] = Field(description="Sorted keys of the subgraph output mapping.")


class LlmUsage(BaseModel):
    """Token counts for one chat-model generation (LangChain ``UsageMetadata`` shape).

    Owned JARL snapshot for ``events.jsonl``. Accepts LangChain
    ``AIMessage.usage_metadata`` via ``from_mapping``. Detail breakdowns
    (cache, reasoning, …) are reserved for a later revision.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(description="Input (prompt) token count.")
    output_tokens: int = Field(description="Output (completion) token count.")
    total_tokens: int = Field(description="Total tokens (input + output).")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> LlmUsage | None:
        """Build usage from a LangChain ``UsageMetadata`` mapping, or ``None`` if incomplete.

        Reads only the standardized keys ``input_tokens``, ``output_tokens``, and
        ``total_tokens``. When ``total_tokens`` is missing, derives it from input
        and output. Returns ``None`` if any of the three cannot be resolved.
        """
        if payload is None:
            return None
        raw_in = payload.get("input_tokens")
        raw_out = payload.get("output_tokens")
        raw_total = payload.get("total_tokens")
        input_tokens = raw_in if isinstance(raw_in, int) and raw_in >= 0 else None
        output_tokens = raw_out if isinstance(raw_out, int) and raw_out >= 0 else None
        total_tokens = raw_total if isinstance(raw_total, int) and raw_total >= 0 else None
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if input_tokens is None or output_tokens is None or total_tokens is None:
            return None
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


class LlmStartEvent(LangGraphNodeStamp):
    """Chat-model generation started inside a LangGraph node."""

    event: Literal["llm_start"] = "llm_start"


class LlmEndEvent(LangGraphNodeStamp):
    """Chat-model generation finished inside a LangGraph node."""

    event: Literal["llm_end"] = "llm_end"
    duration_ms: float | None = Field(default=None, description="Generation duration in milliseconds.")
    usage: LlmUsage | None = Field(
        default=None,
        description="Token usage from ``AIMessage.usage_metadata``, when present.",
    )


class LlmErrorEvent(LangGraphNodeStamp):
    """Chat-model generation failed inside a LangGraph node."""

    event: Literal["llm_error"] = "llm_error"
    error_type: str | None = Field(default=None, description="Python exception type name.")
    error: str | None = Field(default=None, description="Error message without prompt or completion text.")


RunEvent = Annotated[
    ExperimentInvokeEvent
    | ExperimentResultEvent
    | GraphNodeEvent
    | SubgraphInvokeEvent
    | SubgraphResultEvent
    | LlmStartEvent
    | LlmEndEvent
    | LlmErrorEvent,
    Field(discriminator="event"),
]
"""Discriminated union of known ``events.jsonl`` row types."""

_RUN_EVENT_ADAPTER: TypeAdapter[RunEvent] = TypeAdapter(RunEvent)


def parse_run_event(payload: Mapping[str, object]) -> RunEvent:
    """Parse one JSON object into a typed run event.

    Raises:
        ValidationError: If ``event`` is missing or not a known kind.
    """
    return _RUN_EVENT_ADAPTER.validate_python(payload)


def of_kind[TEvent: RunEventBase](events: Sequence[RunEventBase], typ: type[TEvent]) -> tuple[TEvent, ...]:
    """Return the events that are instances of ``typ``, in the original order."""
    return tuple(event for event in events if isinstance(event, typ))


_WINDOW_PART_BY_TYPE: dict[type[RunEventBase], str] = {
    GraphNodeEvent: "graph_nodes",
    LlmStartEvent: "llm_starts",
    LlmEndEvent: "llm_ends",
    LlmErrorEvent: "llm_errors",
}
"""Event types collected into the list fields of one ``RunWindow``."""


@dataclass(frozen=True, slots=True)
class RunWindow:
    """One ``invoke_experiment`` turn as recorded in ``events.jsonl``."""

    invoke: ExperimentInvokeEvent
    result: ExperimentResultEvent | None
    graph_nodes: tuple[GraphNodeEvent, ...]
    llm_starts: tuple[LlmStartEvent, ...] = ()
    llm_ends: tuple[LlmEndEvent, ...] = ()
    llm_errors: tuple[LlmErrorEvent, ...] = ()

    @classmethod
    def from_parts(
        cls,
        invoke: ExperimentInvokeEvent,
        *,
        result: ExperimentResultEvent | None = None,
        graph_nodes: Sequence[GraphNodeEvent] = (),
        llm_starts: Sequence[LlmStartEvent] = (),
        llm_ends: Sequence[LlmEndEvent] = (),
        llm_errors: Sequence[LlmErrorEvent] = (),
    ) -> RunWindow:
        """Build a window from an invoke stamp and the events collected so far."""
        return cls(
            invoke=invoke,
            result=result,
            graph_nodes=tuple(graph_nodes),
            llm_starts=tuple(llm_starts),
            llm_ends=tuple(llm_ends),
            llm_errors=tuple(llm_errors),
        )

    @property
    def experiment_node_id(self) -> str | None:
        """Return the experiment node id, preferring the result stamp."""
        if self.result is not None:
            return self.result.experiment_node_id
        return self.invoke.experiment_node_id

    @property
    def phase(self) -> str:
        """Return the LangGraph phase, preferring the result stamp."""
        if self.result is not None:
            return self.result.phase
        return self.invoke.phase


@dataclass(frozen=True, slots=True)
class RunWindows:
    """Ordered invoke turns grouped from a run log."""

    windows: tuple[RunWindow, ...] = ()

    def __iter__(self) -> Iterator[RunWindow]:
        """Iterate invoke turns in log order."""
        return iter(self.windows)

    def __len__(self) -> int:
        """Return how many invoke turns were grouped."""
        return len(self.windows)

    @classmethod
    def from_events(cls, events: Sequence[RunEventBase]) -> RunWindows:
        """Group invoke, graph-node, result, and LLM events into per-turn windows.

        Subgraph rows are ignored. A new invoke closes an in-flight window.
        Events before the first invoke are skipped.
        """
        windows: list[RunWindow] = []
        invoke: ExperimentInvokeEvent | None = None
        parts: dict[str, list[RunEventBase]] = {}

        def close(*, result: ExperimentResultEvent | None = None) -> None:
            nonlocal invoke, parts
            if invoke is None:
                return
            windows.append(RunWindow.from_parts(invoke, result=result, **parts))
            invoke = None
            parts = {}

        for event in events:
            if isinstance(event, ExperimentInvokeEvent):
                close()
                invoke = event
                parts = {key: [] for key in _WINDOW_PART_BY_TYPE.values()}
                continue
            if invoke is None:
                continue
            key = _WINDOW_PART_BY_TYPE.get(type(event))
            if key is not None:
                parts[key].append(event)
                continue
            if isinstance(event, ExperimentResultEvent):
                close(result=event)
        close()
        return cls(windows=tuple(windows))

    @property
    def last_result(self) -> ExperimentResultEvent | None:
        """Return the last completed invoke result, if any."""
        for window in reversed(self.windows):
            if window.result is not None:
                return window.result
        return None

    @property
    def sticky_experiment_node_id(self) -> str | None:
        """Return the experiment node id from the latest stamp."""
        last_result = self.last_result
        if last_result is not None:
            return last_result.experiment_node_id
        if self.windows:
            return self.windows[-1].invoke.experiment_node_id
        return None

    @property
    def graph_nodes(self) -> tuple[GraphNodeEvent, ...]:
        """Return graph-node events across all turns, in log order."""
        return tuple(node for window in self.windows for node in window.graph_nodes)

    @property
    def last_llm_start(self) -> LlmStartEvent | None:
        """Return the latest ``llm_start`` across all turns, if any."""
        for window in reversed(self.windows):
            if window.llm_starts:
                return window.llm_starts[-1]
        return None

    @property
    def llm_generation_in_flight(self) -> bool:
        """Return whether the open turn has more ``llm_start`` than end/error events.

        True between ``llm_start`` and the matching ``llm_end`` / ``llm_error`` while
        an invoke has not completed. Safe when there are no windows.
        """
        if not self.windows:
            return False
        last = self.windows[-1]
        if last.result is not None:
            return False
        done = len(last.llm_ends) + len(last.llm_errors)
        return len(last.llm_starts) > done

    @property
    def sticky_phase(self) -> str | None:
        """Return the LangGraph node that should label the current view.

        An in-flight invoke (no result yet) prefers the latest ``llm_start``
        in that window, then the latest ``graph_node``, then the invoke phase.
        Completed turns use the result stamp.
        """
        if self.windows and self.windows[-1].result is None:
            last = self.windows[-1]
            if last.llm_starts and last.llm_starts[-1].node:
                return last.llm_starts[-1].node
            if last.graph_nodes:
                return last.graph_nodes[-1].node
            return last.invoke.phase
        last_result = self.last_result
        if last_result is not None:
            return last_result.phase
        graph_nodes = self.graph_nodes
        if graph_nodes:
            return graph_nodes[-1].node
        return None
