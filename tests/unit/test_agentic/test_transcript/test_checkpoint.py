"""Tests for checkpoint transcript reader and active-namespace pointer."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from jarl.agentic.langgraph.compile import (
    build_checkpointer,
    invoke_experiment,
    langgraph_checkpoint_db_path,
)
from jarl.agentic.transcript import (
    ActiveCheckpointPointer,
    CheckpointTranscriptReader,
    ConversationTranscript,
    active_checkpoint_ns_path,
)
from jarl.agentic.workflow import AgenticWorkflow
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel


class TestConversationRevision:
    def test_revision_changes_when_pointer_set(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        reader = CheckpointTranscriptReader.for_experiment(prepared_experiment)
        assert reader is not None
        before = reader.revision()

        ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id).set("operate:x")
        try:
            after = reader.revision()
        finally:
            ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id).clear()

        assert after.token != before.token

    def test_revision_changes_after_invoke_writes_sqlite(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        reader = CheckpointTranscriptReader.for_experiment(prepared_experiment)
        assert reader is not None
        before = reader.revision()

        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        workflow.set_llm(llm)
        invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="hola")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        after = reader.revision()
        assert after.token != before.token
        assert ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id).read() is None


class TestActiveCheckpointPointer:
    def test_set_read_clear_under_run_dir(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        pointer = ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id)

        assert pointer.path == active_checkpoint_ns_path(prepared_experiment, workflow.thread_id)
        assert pointer.read() is None
        pointer.set("operate:abc")
        assert pointer.read() == "operate:abc"
        pointer.clear()
        assert pointer.read() is None
        assert not pointer.path.is_file()

    def test_context_manager_clears_stale_and_on_exit(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        pointer = ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id)
        pointer.set("operate:stale")

        with pointer as sink:
            assert sink.read() is None
            sink.set("operate:live")
            assert sink.read() == "operate:live"

        assert pointer.read() is None
        assert not pointer.path.is_file()

    def test_threads_do_not_share_pointer_files(self, prepared_experiment: Path) -> None:
        first = ActiveCheckpointPointer.for_thread(prepared_experiment, "thread_a")
        second = ActiveCheckpointPointer.for_thread(prepared_experiment, "thread_b")
        first.set("operate:a")
        second.set("operate:b")

        assert first.read() == "operate:a"
        assert second.read() == "operate:b"
        assert first.path != second.path


class TestLoadConversationTranscript:
    def test_returns_none_without_session(self, tmp_path: Path) -> None:
        assert ConversationTranscript.load(tmp_path) is None

    def test_loads_messages_from_parent_checkpoint(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="respuesta lista")]))
        workflow.set_llm(llm)
        invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="pregunta inicial")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        transcript = ConversationTranscript.load(prepared_experiment)

        assert isinstance(transcript, ConversationTranscript)
        assert transcript.thread_id == workflow.thread_id
        assert transcript.checkpoint_ns == ""
        assert len(transcript.messages) == 2
        assert transcript.messages[0].role == "human"
        assert transcript.messages[0].content == "pregunta inicial"
        assert transcript.messages[1].role == "ai"
        assert transcript.messages[1].content == "respuesta lista"

    def test_returns_empty_messages_when_checkpoint_missing(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        transcript = ConversationTranscript.load(prepared_experiment)

        assert transcript is not None
        assert transcript.thread_id == workflow.thread_id
        assert transcript.messages == ()

    def test_clears_pointer_after_invoke(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        workflow.set_llm(llm)
        invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="hola")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        assert ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id).read() is None


class TestChildCheckpointNamespace:
    def test_explicit_checkpoint_ns_loads_child_messages(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="voy",
                        tool_calls=[{"name": "graph_summary", "args": {}, "id": "c1", "type": "tool_call"}],
                    ),
                    AIMessage(content="fin"),
                ]
            )
        )
        workflow.set_llm(llm)
        invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="hola")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        child_ns = _first_nested_namespace(prepared_experiment, workflow.thread_id)
        assert child_ns is not None

        transcript = ConversationTranscript.load(prepared_experiment, checkpoint_ns=child_ns)

        assert transcript is not None
        assert transcript.checkpoint_ns == child_ns
        assert len(transcript.messages) >= 4
        assert transcript.messages[0].content == "hola"
        assert any(message.content == "fin" for message in transcript.messages if message.role == "ai")

    def test_pointer_selects_child_namespace(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="voy",
                        tool_calls=[{"name": "graph_summary", "args": {}, "id": "c1", "type": "tool_call"}],
                    ),
                    AIMessage(content="fin"),
                ]
            )
        )
        workflow.set_llm(llm)
        invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="hola")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )
        child_ns = _first_nested_namespace(prepared_experiment, workflow.thread_id)
        assert child_ns is not None

        pointer = ActiveCheckpointPointer.for_thread(prepared_experiment, workflow.thread_id)
        pointer.set(child_ns)
        try:
            transcript = ConversationTranscript.load(prepared_experiment)
        finally:
            pointer.clear()

        assert transcript is not None
        assert transcript.checkpoint_ns == child_ns
        assert len(transcript.messages) >= 4

    def test_readonly_load_during_invoke_does_not_raise(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="voy",
                        tool_calls=[{"name": "graph_summary", "args": {}, "id": "c1", "type": "tool_call"}],
                    ),
                    AIMessage(content="fin"),
                ]
            )
        )
        workflow.set_llm(llm)

        errors: list[str] = []
        samples: list[int] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    transcript = ConversationTranscript.load(prepared_experiment)
                    if transcript is not None:
                        samples.append(len(transcript.messages))
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                time.sleep(0.05)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            time.sleep(0.05)
            invoke_experiment(
                workflow,
                input_state={"messages": [HumanMessage(content="hola")]},
                checkpointer=build_checkpointer(prepared_experiment),
            )
        finally:
            stop.set()
            thread.join(timeout=2)

        assert errors == []
        assert samples
        reader_obj = CheckpointTranscriptReader.for_experiment(prepared_experiment)
        assert reader_obj is not None
        parent = reader_obj.load_parent()
        assert len(parent.messages) >= 2


def _first_nested_namespace(exp_dir: Path, thread_id: str) -> str | None:
    db_path = langgraph_checkpoint_db_path(exp_dir)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ? AND checkpoint_ns != ''",
            (thread_id,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    return str(rows[0][0])
