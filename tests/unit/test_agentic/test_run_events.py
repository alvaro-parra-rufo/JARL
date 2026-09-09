"""Tests for typed run-log events."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarl.agentic.run_events import (
    ExperimentInvokeEvent,
    ExperimentResultEvent,
    GraphNodeEvent,
    LlmEndEvent,
    LlmErrorEvent,
    LlmStartEvent,
    LlmUsage,
    RunWindows,
    of_kind,
    parse_run_event,
)


class TestLlmUsage:
    def test_from_mapping_accepts_langchain_shape(self) -> None:
        usage = LlmUsage.from_mapping({"input_tokens": 10, "output_tokens": 32, "total_tokens": 42})

        assert usage == LlmUsage(input_tokens=10, output_tokens=32, total_tokens=42)

    def test_from_mapping_derives_total(self) -> None:
        usage = LlmUsage.from_mapping({"input_tokens": 4, "output_tokens": 6})

        assert usage == LlmUsage(input_tokens=4, output_tokens=6, total_tokens=10)

    def test_from_mapping_rejects_incomplete(self) -> None:
        assert LlmUsage.from_mapping(None) is None
        assert LlmUsage.from_mapping({"input_tokens": 1}) is None
        assert LlmUsage.from_mapping({"total_tokens": 9}) is None


class TestParseRunEvent:
    def test_parses_graph_node(self) -> None:
        event = parse_run_event(
            {"event": "graph_node", "ts": "2026-08-19T00:00:00+00:00", "node": "operate", "message_count": 3}
        )

        assert event == GraphNodeEvent(
            ts="2026-08-19T00:00:00+00:00",
            node="operate",
            message_count=3,
        )

    def test_parses_experiment_invoke_without_node_id(self) -> None:
        event = parse_run_event(
            {
                "event": "experiment_invoke",
                "ts": "2026-08-19T00:00:00+00:00",
                "phase": "setup",
                "message_count": 1,
            }
        )

        assert isinstance(event, ExperimentInvokeEvent)
        assert event.experiment_node_id is None
        assert event.phase == "setup"

    def test_parses_llm_end(self) -> None:
        event = parse_run_event(
            {
                "event": "llm_end",
                "ts": "2026-08-19T00:00:00+00:00",
                "node": "operate",
                "duration_ms": 1200.0,
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 300,
                    "total_tokens": 800,
                },
            }
        )

        assert event == LlmEndEvent(
            ts="2026-08-19T00:00:00+00:00",
            node="operate",
            duration_ms=1200.0,
            usage=LlmUsage(input_tokens=500, output_tokens=300, total_tokens=800),
        )

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValidationError):
            parse_run_event({"event": "not_a_kind", "ts": "2026-08-19T00:00:00+00:00"})


class TestOfKind:
    def test_filters_in_original_order(self) -> None:
        events = (
            ExperimentInvokeEvent(phase="operate", message_count=1),
            GraphNodeEvent(node="setup", message_count=2),
            GraphNodeEvent(node="operate", message_count=4),
        )

        nodes = of_kind(events, GraphNodeEvent)

        assert [event.node for event in nodes] == ["setup", "operate"]


class TestRunWindows:
    def test_groups_invoke_nodes_and_result(self) -> None:
        invoke = ExperimentInvokeEvent(phase="setup", message_count=1)
        setup = GraphNodeEvent(node="setup", message_count=4)
        operate = GraphNodeEvent(node="operate", message_count=5)
        result = ExperimentResultEvent(
            phase="operate",
            experiment_node_id="n1",
            message_count=5,
            nodes_visited=["setup", "operate"],
        )
        llm = LlmEndEvent(node="setup", duration_ms=1.0)

        windows = RunWindows.from_events((invoke, setup, llm, operate, result))

        window = next(iter(windows))
        assert len(windows) == 1
        assert window.graph_nodes == (setup, operate)
        assert window.llm_ends == (llm,)
        assert window.result == result
        assert window.experiment_node_id == "n1"
        assert window.phase == "operate"
        assert windows.last_result == result
        assert windows.sticky_phase == "operate"
        assert windows.graph_nodes == (setup, operate)

    def test_keeps_in_flight_invoke_without_result(self) -> None:
        invoke = ExperimentInvokeEvent(phase="setup", message_count=1)
        node = GraphNodeEvent(node="setup", message_count=2)

        windows = RunWindows.from_events((invoke, node))

        assert len(windows) == 1
        assert windows.windows[0].result is None
        assert windows.last_result is None
        assert windows.sticky_experiment_node_id is None
        assert windows.sticky_phase == "setup"

    def test_new_invoke_closes_previous_window(self) -> None:
        first = ExperimentInvokeEvent(phase="setup", message_count=1)
        second = ExperimentInvokeEvent(phase="operate", message_count=3, experiment_node_id="n1")

        windows = RunWindows.from_events((first, second))

        assert len(windows) == 2
        assert windows.windows[0].result is None
        assert windows.sticky_experiment_node_id == "n1"
        assert windows.sticky_phase == "operate"

    def test_in_flight_prefers_last_llm_start(self) -> None:
        invoke = ExperimentInvokeEvent(phase="setup", message_count=1)
        start = LlmStartEvent(node="setup")
        error = LlmErrorEvent(node="setup", error_type="TimeoutError", error="timed out")

        windows = RunWindows.from_events((invoke, start, error))

        assert windows.sticky_phase == "setup"
        assert windows.last_llm_start == start
        assert windows.windows[0].llm_starts == (start,)
        assert windows.windows[0].llm_errors == (error,)

    def test_llm_generation_in_flight(self) -> None:
        invoke = ExperimentInvokeEvent(phase="operate", message_count=1)
        start = LlmStartEvent(node="operate")
        end = LlmEndEvent(
            node="operate",
            duration_ms=10.0,
            usage=LlmUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )

        open_windows = RunWindows.from_events((invoke, start))
        assert open_windows.llm_generation_in_flight is True

        between_tools = RunWindows.from_events((invoke, start, end))
        assert between_tools.llm_generation_in_flight is False

        start2 = LlmStartEvent(node="operate")
        second_open = RunWindows.from_events((invoke, start, end, start2))
        assert second_open.llm_generation_in_flight is True

        second_done = RunWindows.from_events(
            (
                invoke,
                start,
                end,
                start2,
                LlmEndEvent(
                    node="operate",
                    duration_ms=5.0,
                    usage=LlmUsage(input_tokens=1, output_tokens=0, total_tokens=1),
                ),
            )
        )
        assert second_done.llm_generation_in_flight is False

        completed = RunWindows.from_events(
            (
                invoke,
                start,
                end,
                ExperimentResultEvent(phase="operate", message_count=2, nodes_visited=["operate"]),
            )
        )
        assert completed.llm_generation_in_flight is False
