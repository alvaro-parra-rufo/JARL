"""Tests for experiment LangGraph `AgentState`."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from jarl.agentic.langgraph.state import AgentState


def _state(*, messages: list[object] | None = None, thread_id: str = "thread") -> AgentState:
    return AgentState(
        messages=list(messages or []),
        experiment_dir="/exp",
        thread_id=thread_id,
    )


class TestAgentState:
    def test_from_mapping_wraps_invoke_payload(self) -> None:
        human = HumanMessage(content="hola")

        state = AgentState.from_mapping(
            {
                "messages": [human],
                "experiment_dir": "/exp",
                "thread_id": "t1",
            }
        )

        assert state.message_count == 1
        assert state.messages == [human]
        assert state.experiment_dir == "/exp"
        assert state.thread_id == "t1"

    def test_from_mapping_returns_existing_instance(self) -> None:
        original = _state(messages=[HumanMessage(content="hola")])

        assert AgentState.from_mapping(original) is original

    def test_to_payload_is_shallow_channel_mapping(self) -> None:
        human = HumanMessage(content="hola")
        state = _state(messages=[human], thread_id="t1")

        payload = state.to_payload()

        assert payload["messages"] is state.messages
        assert payload["experiment_dir"] == "/exp"
        assert payload["thread_id"] == "t1"

    def test_tool_called_looks_only_after_offset(self) -> None:
        prior_finish = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "session_finish",
                    "args": {},
                    "id": "old",
                    "type": "tool_call",
                }
            ],
        )
        later_summary = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "graph_summary",
                    "args": {},
                    "id": "new",
                    "type": "tool_call",
                }
            ],
        )
        state = _state(messages=[prior_finish, later_summary])

        assert state.tool_called("session_finish") is True
        assert state.tool_called("session_finish", after=1) is False
        assert state.tool_called("graph_summary", after=1) is True

    def test_add_messages_reducer_appends_instead_of_replacing(self) -> None:
        graph = StateGraph(AgentState)

        def append_reply(_state: AgentState) -> dict[str, object]:
            return {"messages": [AIMessage(content="ok")]}

        graph.add_node("reply", append_reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        compiled = graph.compile()

        result = AgentState.from_mapping(compiled.invoke(_state(messages=[HumanMessage(content="hola")]).to_payload()))

        assert [message.content for message in result.messages] == ["hola", "ok"]
        assert result.message_count == 2
