"""Tests for the core model-debug timeline builder."""

from __future__ import annotations

import inspect
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

import jarl.agentic.debug as debug_mod
from jarl.agentic.debug import (
    DebugContext,
    DebugSegment,
    ModelDebugView,
    attach_segment_context,
    segment_transcript,
)
from jarl.agentic.langgraph.compile import build_checkpointer, compile_experiment_graph
from jarl.agentic.run_events import (
    ExperimentInvokeEvent,
    ExperimentResultEvent,
    GraphNodeEvent,
    LlmEndEvent,
    LlmErrorEvent,
    LlmStartEvent,
    LlmUsage,
    RunWindows,
)
from jarl.agentic.session import load_session
from jarl.agentic.transcript import ConversationTranscript, TranscriptMessage
from jarl.agentic.workflow import AgenticWorkflow
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel


class TestBuildModelDebugView:
    def test_cuts_by_graph_node_not_tool_name(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "graph_create_root",
                                "args": {"label": "baseline"},
                                "id": "call_create_root",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="setup done"),
                    AIMessage(content="operate ok"),
                ]
            )
        )
        workflow.set_llm(llm)
        workflow.invoke({"messages": [HumanMessage(content="inicializa")]})

        view = ModelDebugView.from_workflow(workflow)

        assert [segment.context.node for segment in view.segments] == ["setup", "operate"]
        assert not view.segments[0].context.inferred
        tool_names = [message.tool_name for message in view.segments[0].messages if message.role == "tool"]
        assert "graph_create_root" in tool_names
        assert all(segment.context.node != "graph_create_root" for segment in view.segments)

    def test_prepared_experiment_uses_operate_segments(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="listo")]))
        workflow.set_llm(llm)
        workflow.invoke({"messages": [HumanMessage(content="resume")]})

        view = ModelDebugView.from_workflow(workflow)

        assert [segment.context.node for segment in view.segments] == ["operate"]
        assert view.phase == "operate"
        assert view.experiment_node_id == workflow.try_current_node_id()

    def test_missing_graph_node_events_yields_inferred_segment(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
        )

        view = ModelDebugView.from_workflow(workflow)

        assert len(view.segments) == 1
        assert view.segments[0].context.inferred is True
        assert view.segments[0].context.node == "operate"

    def test_excluded_finish_is_absent_from_segment_tools(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        workflow.set_llm(llm)
        workflow.invoke(
            {"messages": [HumanMessage(content="resume")]},
            filter_by={"exclude_labels": {"finish"}},
        )

        view = ModelDebugView.from_workflow(workflow)
        tool_names = {tool.name for segment in view.segments for tool in segment.context.tools}

        assert "session_finish" not in tool_names
        assert "graph_fork" in tool_names

    def test_running_does_not_slice_by_checkpoint_counts(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "graph_create_root",
                                "args": {"label": "baseline"},
                                "id": "call_create_root",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="setup done"),
                    AIMessage(content="operate ok"),
                ]
            )
        )
        workflow.set_llm(llm)
        workflow.invoke({"messages": [HumanMessage(content="inicializa")]})

        finished = ModelDebugView.from_workflow(workflow, running=False)
        live = ModelDebugView.from_workflow(workflow, running=True)

        assert [segment.context.node for segment in finished.segments] == ["setup", "operate"]
        assert len(live.segments) == 1
        assert live.running is True
        assert live.segments[0].context.node == "operate"
        assert live.segments[0].messages
        # Live prefers progress/audit; do not equate its length to checkpoint slices.

    def test_segment_attaches_llm_generations(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        workflow.set_llm(llm)
        workflow.invoke({"messages": [HumanMessage(content="resume")]})

        view = ModelDebugView.from_workflow(workflow)
        generations = view.segments[0].context.generations

        assert view.segments[0].context.node == "operate"
        assert generations
        assert all(event.node == "operate" for event in generations)
        assert all(event.event == "llm_end" for event in generations)

    def test_system_prompt_comes_from_compiled_nodes(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        workflow.set_llm(llm)
        workflow.invoke({"messages": [HumanMessage(content="resume")]})

        view = ModelDebugView.from_workflow(workflow)
        session = load_session(prepared_experiment)

        assert session is not None
        assert "jarl.agentic.prompts" not in inspect.getsource(debug_mod)
        assert view.segments[0].context.system_prompt == session.compiled_nodes["operate"].system_prompt
        assert view.segments[0].context.system_prompt is not None
        via_path = ModelDebugView.from_experiment(prepared_experiment)
        assert via_path.phase == view.phase
        assert via_path.experiment_node_id == view.experiment_node_id

    def test_from_experiment_without_session_is_empty(self, tmp_path: Path) -> None:
        view = ModelDebugView.from_experiment(tmp_path)

        assert view.segments == ()
        assert view.phase is None


class TestSegmentTranscriptRunning:
    def test_running_keeps_all_messages_in_one_segment(self, tmp_path: Path) -> None:
        transcript = ConversationTranscript(
            experiment_dir=tmp_path,
            thread_id="thread-1",
            messages=(
                TranscriptMessage(index=0, role="human", content="hola"),
                TranscriptMessage(index=1, role="ai", content="ok"),
            ),
        )

        segments = segment_transcript(transcript, RunWindows(), running=True)

        assert len(segments) == 1
        assert segments[0].context.inferred is True
        assert len(segments[0].messages) == 2


class TestAttachSegmentContext:
    def test_running_uses_last_llm_start_node(self, tmp_path: Path) -> None:
        transcript = ConversationTranscript(
            experiment_dir=tmp_path,
            thread_id="thread-1",
            messages=(TranscriptMessage(index=0, role="human", content="hola"),),
        )
        windows = RunWindows.from_events(
            (
                ExperimentInvokeEvent(phase="setup", message_count=1),
                LlmStartEvent(node="setup"),
            )
        )

        segments = segment_transcript(transcript, windows, running=True)

        assert len(segments) == 1
        assert segments[0].context.node == "setup"
        assert segments[0].context.inferred is False

    def test_attaches_generations_and_errors_for_matching_node(self) -> None:
        generation = LlmEndEvent(
            node="operate",
            duration_ms=1200.0,
            usage=LlmUsage(input_tokens=500, output_tokens=300, total_tokens=800),
        )
        error = LlmErrorEvent(node="operate", error_type="TimeoutError", error="timed out")
        segments = (
            DebugSegment(
                context=DebugContext.placeholder("operate", experiment_node_id="n1", inferred=False),
                messages=(TranscriptMessage(index=0, role="ai", content="ok"),),
            ),
        )
        windows = RunWindows.from_events(
            (
                ExperimentInvokeEvent(phase="operate", message_count=0, experiment_node_id="n1"),
                generation,
                error,
                GraphNodeEvent(node="operate", message_count=1),
                ExperimentResultEvent(
                    phase="operate",
                    experiment_node_id="n1",
                    message_count=1,
                    nodes_visited=["operate"],
                ),
            )
        )

        attached = attach_segment_context(segments, {}, windows=windows)

        assert attached[0].context.generations == (generation,)
        assert attached[0].context.errors == (error,)
