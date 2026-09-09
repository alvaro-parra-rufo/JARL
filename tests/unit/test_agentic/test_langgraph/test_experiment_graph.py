"""Tests for the experiment LangGraph runtime."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pytest_mock import MockerFixture

from jarl.agentic.audit import load_run_events, run_dir
from jarl.agentic.errors import LangGraphNotConfiguredError
from jarl.agentic.langgraph.compile import (
    build_checkpointer,
    build_langchain_tools,
    compile_experiment_graph,
    invoke_experiment,
    langgraph_checkpoint_db_path,
    langgraph_checkpoint_dir,
)
from jarl.agentic.langgraph.graphs.experiment import (
    OPERATE_INCLUDE_LABELS,
    SETUP_INCLUDE_LABELS,
    resolve_experiment_phase,
    route_after_setup,
)
from jarl.agentic.langgraph.state import AgentState
from jarl.agentic.prompts import OPERATE_SYSTEM_PROMPT, SETUP_SYSTEM_PROMPT
from jarl.agentic.run_events import (
    ExperimentInvokeEvent,
    ExperimentResultEvent,
    GraphNodeEvent,
    LlmEndEvent,
    of_kind,
)
from jarl.agentic.session import load_session
from jarl.agentic.tools.registry import REGISTRY
from jarl.agentic.workflow import AgenticWorkflow
from tests.helpers.fake_chat_model import RecordingFakeChatModel, ToolBindingFakeChatModel


class TestExperimentPhaseRouting:
    def test_resolve_setup_for_empty_experiment(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)

        assert resolve_experiment_phase(workflow) == "setup"

    def test_resolve_operate_for_prepared_experiment(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)

        assert resolve_experiment_phase(workflow) == "operate"


class TestPostSetupRouting:
    def test_route_after_setup_ends_when_experiment_stays_empty(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        state = AgentState(
            messages=[HumanMessage(content="hola")],
            experiment_dir=str(empty_experiment),
            thread_id=workflow.thread_id,
        )

        assert route_after_setup(workflow, state) == "end"

    def test_route_after_setup_operates_when_nodes_exist(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        state = AgentState(
            messages=[HumanMessage(content="hola")],
            experiment_dir=str(prepared_experiment),
            thread_id=workflow.thread_id,
        )

        assert route_after_setup(workflow, state) == "operate"


class TestExperimentGraphExecution:
    def _streamed_nodes(
        self,
        workflow: AgenticWorkflow,
        llm: ToolBindingFakeChatModel,
        *,
        experiment_dir: Path,
        messages: list[HumanMessage],
    ) -> list[str]:
        compiled = compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(experiment_dir),
        )
        state = AgentState(
            messages=messages,
            experiment_dir=str(experiment_dir),
            thread_id=workflow.thread_id,
        )
        config = {"configurable": {"thread_id": workflow.thread_id}}
        return [
            next(iter(update)) for update in compiled.stream(state.to_payload(), config=config, stream_mode="updates")
        ]

    def test_prepared_experiment_routes_bootstrap_to_operate(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="operate ok")]))

        nodes = self._streamed_nodes(
            workflow,
            llm,
            experiment_dir=prepared_experiment,
            messages=[HumanMessage(content="resume")],
        )

        assert nodes == ["bootstrap", "operate"]

    def test_setup_chains_to_operate_after_create_root(
        self,
        empty_experiment: Path,
    ) -> None:
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

        nodes = self._streamed_nodes(
            workflow,
            llm,
            experiment_dir=empty_experiment,
            messages=[HumanMessage(content="inicializa el experimento")],
        )

        assert nodes == ["bootstrap", "setup", "operate"]
        assert resolve_experiment_phase(workflow) == "operate"

    def test_setup_ends_when_experiment_stays_empty(self, empty_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="solo setup")]))

        nodes = self._streamed_nodes(
            workflow,
            llm,
            experiment_dir=empty_experiment,
            messages=[HumanMessage(content="hola")],
        )

        assert nodes == ["bootstrap", "setup"]
        assert resolve_experiment_phase(workflow) == "setup"


class TestExperimentToolFiltering:
    def test_setup_tools_include_create_root_and_exclude_train(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        setup_names = {
            tool.name
            for tool in build_langchain_tools(
                REGISTRY.filter_by(include_labels=set(SETUP_INCLUDE_LABELS)),
                workflow,
            )
        }
        operate_names = {
            tool.name
            for tool in build_langchain_tools(
                REGISTRY.filter_by(include_labels=set(OPERATE_INCLUDE_LABELS)),
                workflow,
            )
        }

        assert "graph_create_root" in setup_names
        assert "session_status" in setup_names
        assert "session_finish" in setup_names
        assert "graph_fork" not in setup_names
        assert "train_run" not in setup_names
        assert "train_run" in operate_names
        assert "graph_create_root" not in operate_names
        assert "graph_fork" in operate_names
        assert "graph_reward" in operate_names
        assert "graph_set_reward" in operate_names
        assert "graph_reward" not in setup_names
        assert "graph_set_reward" not in setup_names
        assert "session_finish" in operate_names


class TestDriverToolFilter:
    def test_operate_includes_session_finish_without_driver_filter(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        captured = self._capture_create_agent_tool_names(mocker)
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
        )

        assert "session_finish" in captured["operate"]
        assert "session_finish" in captured["setup"]

    def test_exclude_finish_hides_session_finish_after_phase_filter(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        captured = self._capture_create_agent_tool_names(mocker)
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
            filter_by={"exclude_labels": {"finish"}},
        )

        assert "session_finish" not in captured["operate"]
        assert "session_finish" not in captured["setup"]
        assert "session_status" in captured["operate"]
        assert "graph_fork" in captured["operate"]

    def _capture_create_agent_tool_names(self, mocker: MockerFixture) -> dict[str, set[str]]:
        from langchain.agents import create_agent as real_create_agent

        captured: dict[str, set[str]] = {}

        def capturing_create_agent(llm: object, tools: object, **kwargs: object) -> object:
            name = kwargs["name"]
            assert isinstance(name, str)
            assert isinstance(tools, list)
            captured[name] = {tool.name for tool in tools}
            return real_create_agent(llm, tools, **kwargs)

        mocker.patch(
            "jarl.agentic.langgraph.graphs.experiment.create_agent",
            side_effect=capturing_create_agent,
        )
        return captured


class TestExperimentSystemPrompts:
    def test_operate_phase_passes_operate_system_prompt_to_llm(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = RecordingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compiled = compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
        )
        compiled.invoke(
            {
                "messages": [HumanMessage(content="resume")],
                "experiment_dir": str(prepared_experiment),
                "thread_id": workflow.thread_id,
            },
            config={"configurable": {"thread_id": workflow.thread_id}},
        )

        assert llm.captured_message_batches
        first_batch = llm.captured_message_batches[0]
        system_messages = [message for message in first_batch if isinstance(message, SystemMessage)]
        assert system_messages
        assert system_messages[0].content == OPERATE_SYSTEM_PROMPT

    def test_setup_phase_passes_setup_system_prompt_to_llm(
        self,
        empty_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(empty_experiment)
        llm = RecordingFakeChatModel(messages=iter([AIMessage(content="hola")]))

        compiled = compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(empty_experiment),
        )
        compiled.invoke(
            {
                "messages": [HumanMessage(content="hola")],
                "experiment_dir": str(empty_experiment),
                "thread_id": workflow.thread_id,
            },
            config={"configurable": {"thread_id": workflow.thread_id}},
        )

        assert llm.captured_message_batches
        first_batch = llm.captured_message_batches[0]
        system_messages = [message for message in first_batch if isinstance(message, SystemMessage)]
        assert system_messages
        assert system_messages[0].content == SETUP_SYSTEM_PROMPT


class TestExperimentGraphCompile:
    def test_compile_experiment_graph_builds(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compiled = compile_experiment_graph(workflow, llm)

        assert compiled is not None
        assert langgraph_checkpoint_dir(prepared_experiment).is_dir()

    def test_compile_persists_create_agent_prompts_on_compiled_nodes(
        self,
        prepared_experiment: Path,
        mocker: MockerFixture,
    ) -> None:
        from langchain.agents import create_agent as real_create_agent

        captured: dict[str, str] = {}

        def capturing_create_agent(*args: object, **kwargs: object) -> object:
            name = kwargs["name"]
            prompt = kwargs["system_prompt"]
            assert isinstance(name, str)
            assert isinstance(prompt, str)
            captured[name] = prompt
            return real_create_agent(*args, **kwargs)

        mocker.patch(
            "jarl.agentic.langgraph.graphs.experiment.create_agent",
            side_effect=capturing_create_agent,
        )
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
        )

        session = load_session(prepared_experiment)
        assert session is not None
        assert set(session.compiled_nodes) == set(captured)
        for name, prompt in captured.items():
            assert session.compiled_nodes[name].system_prompt == prompt

    def test_compile_persists_bind_filter_and_tool_names(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
        )

        session = load_session(prepared_experiment)
        assert session is not None
        setup = session.compiled_nodes["setup"]
        operate = session.compiled_nodes["operate"]
        assert setup.filter_by == {"include_labels": ["setup"]}
        assert operate.filter_by == {"include_labels": ["operate"]}
        assert "session_finish" in setup.tool_names
        assert "session_finish" in operate.tool_names
        assert "graph_create_root" in setup.tool_names
        assert "graph_create_root" not in operate.tool_names
        assert "graph_fork" in operate.tool_names
        assert "graph_fork" not in setup.tool_names

    def test_compile_persists_driver_exclude_labels_on_both_nodes(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
            filter_by={"exclude_labels": {"finish"}},
        )

        session = load_session(prepared_experiment)
        assert session is not None
        for name in ("setup", "operate"):
            info = session.compiled_nodes[name]
            assert info.filter_by["exclude_labels"] == ["finish"]
            assert "session_finish" not in info.tool_names
            assert name in info.filter_by["include_labels"]

    def test_compile_persists_objective_on_compiled_nodes(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))
        objective = "Inspect a demo tree without changing its graph."

        compile_experiment_graph(
            workflow,
            llm,
            checkpointer=build_checkpointer(prepared_experiment),
            objective=objective,
        )

        session = load_session(prepared_experiment)
        assert session is not None
        for info in session.compiled_nodes.values():
            assert objective in info.system_prompt
            assert "session_finish" in info.system_prompt
            assert "non-interactive" in info.system_prompt.lower()


class TestSqliteCheckpointer:
    def test_build_checkpointer_creates_sqlite_database(self, prepared_experiment: Path) -> None:
        build_checkpointer(prepared_experiment)

        assert langgraph_checkpoint_db_path(prepared_experiment).is_file()

    def test_sqlite_checkpointer_persists_thread_state_across_connections(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        first_llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="first turn")]))
        workflow.set_llm(first_llm)
        invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="hola")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        db_path = langgraph_checkpoint_db_path(prepared_experiment)
        assert db_path.is_file()

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        fresh_saver = SqliteSaver(conn)
        fresh_saver.setup()
        second_llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="second turn")]))
        compiled = compile_experiment_graph(workflow, second_llm, checkpointer=fresh_saver)
        config = {"configurable": {"thread_id": workflow.thread_id}}
        snapshot = compiled.get_state(config)
        prior = AgentState.from_mapping(snapshot.values)

        assert any(message.content == "first turn" for message in prior.messages)

        result = AgentState.from_mapping(
            compiled.invoke(
                {
                    "messages": [HumanMessage(content="sigue")],
                    "experiment_dir": str(prepared_experiment),
                    "thread_id": workflow.thread_id,
                },
                config=config,
            )
        )

        assert any(message.content == "first turn" for message in result.messages)
        assert any(message.content == "second turn" for message in result.messages)


class TestExperimentInvoke:
    def test_invoke_requires_llm(self, prepared_experiment: Path, mocker: MockerFixture) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        mocker.patch(
            "jarl.agentic.workflow.load_llm_catalog",
            side_effect=RuntimeError("catalog unavailable"),
        )

        with pytest.raises(LangGraphNotConfiguredError, match="LLM instance or LLM catalog"):
            workflow.invoke({})

    def test_invoke_returns_messages_with_fake_llm(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="listo")]))
        workflow.set_llm(llm)

        result = invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="resume el experimento")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        assert result.thread_id == workflow.thread_id
        assert str(prepared_experiment) == result.experiment_dir
        assert any(message.content == "listo" for message in result.messages)

    def test_invoke_records_agent_run(self, prepared_experiment: Path) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="ok")]))

        workflow.invoke(
            {"messages": [HumanMessage(content="hola")]},
            llm=llm,
        )

        manifest = json.loads((run_dir(prepared_experiment, workflow.thread_id) / "manifest.json").read_text())
        assert manifest["run_kind"] == "agent"
        assert manifest["graph_id"] == "experiment"
        assert manifest["status"] == "completed"


class TestGraphNodeEvents:
    def test_empty_create_root_emits_setup_then_operate(
        self,
        empty_experiment: Path,
    ) -> None:
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

        result = workflow.invoke(
            {"messages": [HumanMessage(content="inicializa el experimento")]},
        )

        events = load_run_events(empty_experiment, workflow.thread_id)
        graph_nodes = of_kind(events, GraphNodeEvent)
        result_event = of_kind(events, ExperimentResultEvent)[0]
        invoke_event = of_kind(events, ExperimentInvokeEvent)[0]
        counts = [event.message_count for event in graph_nodes]

        assert isinstance(result, AgentState)
        assert [event.node for event in graph_nodes] == ["setup", "operate"]
        assert result_event.nodes_visited == ["setup", "operate"]
        assert result_event.phase == "operate"
        assert result_event.experiment_node_id is not None
        assert invoke_event.phase == "setup"
        assert invoke_event.experiment_node_id is None
        assert counts == sorted(counts)
        assert counts[0] > invoke_event.message_count
        assert counts[-1] == result.message_count

    def test_prepared_experiment_emits_only_operate(
        self,
        prepared_experiment: Path,
    ) -> None:
        workflow = AgenticWorkflow.from_experiment(prepared_experiment)
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="listo")]))
        workflow.set_llm(llm)

        result = invoke_experiment(
            workflow,
            input_state={"messages": [HumanMessage(content="resume el experimento")]},
            checkpointer=build_checkpointer(prepared_experiment),
        )

        events = load_run_events(prepared_experiment, workflow.thread_id)
        graph_nodes = of_kind(events, GraphNodeEvent)
        result_event = of_kind(events, ExperimentResultEvent)[0]

        assert isinstance(result, AgentState)
        assert [event.node for event in graph_nodes] == ["operate"]
        assert result_event.nodes_visited == ["operate"]
        assert result_event.phase == "operate"
        assert result_event.experiment_node_id == workflow.try_current_node_id()
        assert graph_nodes[0].message_count == result.message_count
        assert of_kind(events, LlmEndEvent)
        assert all(event.node == "operate" for event in of_kind(events, LlmEndEvent))
