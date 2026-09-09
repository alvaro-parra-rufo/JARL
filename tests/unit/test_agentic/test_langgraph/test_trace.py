"""Tests for LangChain LLM metadata callbacks."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, override
from uuid import uuid4

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

import jarl.agentic.langgraph.trace as trace_mod
from jarl.agentic.audit import RUN_EVENTS_FILENAME, load_run_events, open_run, run_dir
from jarl.agentic.langgraph.trace import LlmGenerationCallback
from jarl.agentic.run_events import LlmEndEvent, LlmErrorEvent, LlmStartEvent, LlmUsage, of_kind
from jarl.agentic.workflow import AgenticWorkflow
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel

_PROMPT_MARKER = "PROMPT_SHOULD_NOT_PERSIST"
_COMPLETION_MARKER = "COMPLETION_SHOULD_NOT_PERSIST"


class _ListTracker:
    """Minimal tracker that captures typed run events in memory."""

    def __init__(self) -> None:
        self.events: list[LlmStartEvent | LlmEndEvent | LlmErrorEvent] = []

    def append_event(self, event: LlmStartEvent | LlmEndEvent | LlmErrorEvent) -> None:
        """Append one event as ``RunTracker.append_event`` would."""
        self.events.append(event)


class _RaisingChatModel(ToolBindingFakeChatModel):
    """Fake chat model that fails after LangChain has started the LLM run."""

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise RuntimeError("provider down")


def _dumped_blob(events: list[LlmStartEvent | LlmEndEvent | LlmErrorEvent]) -> str:
    """Serialize event payloads for content-leak assertions."""
    return json.dumps([event.model_dump() for event in events])


class TestLlmGenerationCallback:
    def test_llm_end_includes_node_without_payloads(self) -> None:
        tracker = _ListTracker()
        callback = LlmGenerationCallback(tracker)  # type: ignore[arg-type]
        run_id = uuid4()

        callback.on_chat_model_start(
            {},
            [[HumanMessage(content=_PROMPT_MARKER)]],
            run_id=run_id,
            metadata={"langgraph_node": "operate"},
        )
        callback.on_llm_end(
            LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content=_COMPLETION_MARKER,
                                usage_metadata={
                                    "input_tokens": 10,
                                    "output_tokens": 32,
                                    "total_tokens": 42,
                                },
                            )
                        )
                    ]
                ],
            ),
            run_id=run_id,
        )

        start, end = tracker.events
        blob = _dumped_blob(list(tracker.events))
        assert start == LlmStartEvent(ts=start.ts, node="operate")
        assert end.node == "operate"
        assert isinstance(end, LlmEndEvent)
        assert end.usage == LlmUsage(input_tokens=10, output_tokens=32, total_tokens=42)
        assert end.duration_ms is not None
        assert "content" not in end.model_dump()
        assert "prompt" not in end.model_dump()
        assert _PROMPT_MARKER not in blob
        assert _COMPLETION_MARKER not in blob

    def test_llm_end_ignores_llm_output_token_usage(self) -> None:
        tracker = _ListTracker()
        callback = LlmGenerationCallback(tracker)  # type: ignore[arg-type]
        run_id = uuid4()
        callback.on_chat_model_start(
            {},
            [[HumanMessage(content=_PROMPT_MARKER)]],
            run_id=run_id,
            metadata={"langgraph_node": "operate"},
        )
        callback.on_llm_end(
            LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content=_COMPLETION_MARKER))]],
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 32,
                        "total_tokens": 42,
                    }
                },
            ),
            run_id=run_id,
        )

        end = tracker.events[1]
        assert isinstance(end, LlmEndEvent)
        assert end.usage is None

    def test_llm_end_uses_top_choice_usage_only(self) -> None:
        tracker = _ListTracker()
        callback = LlmGenerationCallback(tracker)  # type: ignore[arg-type]
        run_id = uuid4()
        callback.on_chat_model_start(
            {},
            [[HumanMessage(content=_PROMPT_MARKER)]],
            run_id=run_id,
            metadata={"langgraph_node": "operate"},
        )
        callback.on_llm_end(
            LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content=_COMPLETION_MARKER,
                                usage_metadata={
                                    "input_tokens": 10,
                                    "output_tokens": 1,
                                    "total_tokens": 11,
                                },
                            )
                        ),
                        ChatGeneration(
                            message=AIMessage(
                                content=_COMPLETION_MARKER,
                                usage_metadata={
                                    "input_tokens": 10,
                                    "output_tokens": 99,
                                    "total_tokens": 109,
                                },
                            )
                        ),
                    ]
                ]
            ),
            run_id=run_id,
        )

        end = tracker.events[1]
        assert isinstance(end, LlmEndEvent)
        assert end.usage == LlmUsage(input_tokens=10, output_tokens=1, total_tokens=11)

    def test_nested_create_agent_uses_parent_agent_name(self) -> None:
        tracker = _ListTracker()
        callback = LlmGenerationCallback(tracker)  # type: ignore[arg-type]
        run_id = uuid4()

        callback.on_chat_model_start(
            {},
            [[HumanMessage(content=_PROMPT_MARKER)]],
            run_id=run_id,
            metadata={
                "langgraph_node": "model",
                "lc_agent_name": "operate",
                "checkpoint_ns": "operate:3ab15528-256d-ecfc-013d-70a3708eb32d",
            },
        )
        callback.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content=_COMPLETION_MARKER))]]),
            run_id=run_id,
        )

        assert tracker.events[0].node == "operate"
        assert tracker.events[1].node == "operate"

    def test_llm_error_records_type_without_payloads(self) -> None:
        tracker = _ListTracker()
        callback = LlmGenerationCallback(tracker)  # type: ignore[arg-type]
        run_id = uuid4()

        callback.on_chat_model_start(
            {},
            [[HumanMessage(content=_PROMPT_MARKER)]],
            run_id=run_id,
            metadata={"langgraph_node": "setup"},
        )
        callback.on_llm_error(RuntimeError("provider down"), run_id=run_id)

        error = tracker.events[1]
        blob = _dumped_blob(list(tracker.events))
        assert isinstance(error, LlmErrorEvent)
        assert error.node == "setup"
        assert error.error_type == "RuntimeError"
        assert error.error == "provider down"
        assert "content" not in error.model_dump()
        assert "prompt" not in error.model_dump()
        assert _PROMPT_MARKER not in blob

    def test_persisted_dump_has_no_message_contents(self, tmp_path: Path) -> None:
        tracker = open_run(tmp_path, "thread-1")
        callback = LlmGenerationCallback(tracker)  # type: ignore[arg-type]
        run_id = uuid4()
        callback.on_chat_model_start(
            {},
            [[HumanMessage(content=_PROMPT_MARKER)]],
            run_id=run_id,
            metadata={"langgraph_node": "operate"},
        )
        callback.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content=_COMPLETION_MARKER))]]),
            run_id=run_id,
        )

        raw = (run_dir(tmp_path, "thread-1") / RUN_EVENTS_FILENAME).read_text(encoding="utf-8")
        events = load_run_events(tmp_path, "thread-1")

        assert of_kind(events, LlmEndEvent)[0].node == "operate"
        assert _PROMPT_MARKER not in raw
        assert _COMPLETION_MARKER not in raw

    def test_module_stays_in_langgraph_runtime(self) -> None:
        source = inspect.getsource(trace_mod)

        assert "jarl.agentic.debug" not in source
        assert "streamlit" not in source
        assert "graph_create_root" not in source
        assert "SETUP_INCLUDE_LABELS" not in source


class TestInvokeRecordsLlmEvents:
    def test_invoke_with_tools_writes_llm_end(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="listo")]))
        workflow.set_llm(llm)

        workflow.invoke({"messages": [HumanMessage(content="resume")]})

        events = load_run_events(prepared_experiment, workflow.thread_id)
        ends = of_kind(events, LlmEndEvent)
        raw = (run_dir(prepared_experiment, workflow.thread_id) / RUN_EVENTS_FILENAME).read_text(encoding="utf-8")

        assert ends
        assert all(event.node == "operate" for event in ends)
        assert "listo" not in raw
        assert "resume" not in raw

    def test_provider_error_writes_llm_error(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        workflow.set_llm(_RaisingChatModel(messages=iter([])))

        with pytest.raises(RuntimeError, match="provider down"):
            workflow.invoke({"messages": [HumanMessage(content=_PROMPT_MARKER)]})

        events = load_run_events(prepared_experiment, workflow.thread_id)
        errors = of_kind(events, LlmErrorEvent)
        raw = (run_dir(prepared_experiment, workflow.thread_id) / RUN_EVENTS_FILENAME).read_text(encoding="utf-8")

        assert errors
        assert errors[0].error_type == "RuntimeError"
        assert _PROMPT_MARKER not in raw
