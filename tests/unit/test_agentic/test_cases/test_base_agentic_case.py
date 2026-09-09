"""Tests for agentic case execution and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from pytest_mock import MockerFixture

from jarl.agentic.audit import audit_index_path
from jarl.agentic.cases import (
    DEFAULT_MAX_CONTINUATIONS,
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
    ValidationReport,
)
from jarl.agentic.cases.builtin.graph_read_without_mutation import (
    GraphReadWithoutMutationCase,
)
from jarl.agentic.llm import LLMSettings
from jarl.agentic.progress import load_progress_events
from jarl.agentic.prompts import CONTINUATION_SYSTEM_PROMPT, DEFAULT_NONINTERACTIVE_OBJECTIVE
from jarl.agentic.session import load_session
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.empty_experiment import EmptyExperimentCase
from jarl.experiments.cases.builtin.navix_prepared_root import NavixPreparedRootCase
from tests.helpers.fake_chat_model import RecordingFakeChatModel, ToolBindingFakeChatModel


class CreateRootAgenticCase(BaseAgenticCase):
    def __init__(self, *, max_continuations: int = 0) -> None:
        super().__init__(
            CaseSpec(
                id="create_root_case",
                title="Create root case",
                tags=frozenset({"graph", "setup"}),
            ),
            EmptyExperimentCase(),
            (AgenticTurn("Create a baseline experiment."),),
            max_continuations=max_continuations,
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        report.check(
            run.graph.as_networkx().number_of_nodes() == 1,
            "Expected the workflow to create one root node.",
        )
        report.check(
            any(event.tool == "graph_create_root" for event in run.audit_events),
            "Expected graph_create_root in the audit index.",
        )
        report.check(
            run.experiment_dir == run.workflow.exp_dir,
            "Expected workflow and case experiment directories to match.",
        )


class TwoTurnAgenticCase(BaseAgenticCase):
    def __init__(self) -> None:
        super().__init__(
            CaseSpec(id="two_turn_case", title="Two turn case"),
            NavixPreparedRootCase(),
            (
                AgenticTurn("First turn."),
                AgenticTurn("Second turn."),
            ),
            max_continuations=0,
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        contents = [message.content for message in run.final_state.messages]
        report.check("first answer" in contents, "Missing first turn response.")
        report.check("second answer" in contents, "Missing second turn response.")


class AlwaysFailAgenticCase(BaseAgenticCase):
    def __init__(self, *, max_continuations: int) -> None:
        super().__init__(
            CaseSpec(id="always_fail_case", title="Always fail case"),
            EmptyExperimentCase(),
            (AgenticTurn("Complete the objective."),),
            max_continuations=max_continuations,
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        del run
        report.check(False, "SECRET_FAILURE_TOKEN")


def _tool_call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _nudge_batches(llm: RecordingFakeChatModel) -> list[list[object]]:
    return [
        batch
        for batch in llm.captured_message_batches
        if any(
            isinstance(message, SystemMessage) and CONTINUATION_SYSTEM_PROMPT in str(message.content)
            for message in batch
        )
    ]


def _batches_mention(llm: RecordingFakeChatModel, needle: str) -> bool:
    return any(needle in str(message.content) for batch in llm.captured_message_batches for message in batch)


def _create_root_llm() -> ToolBindingFakeChatModel:
    return ToolBindingFakeChatModel(
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
                AIMessage(content="operate done"),
            ]
        )
    )


class TestAgenticTurn:
    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            AgenticTurn(" ")


class TestBaseAgenticCase:
    def test_rejects_case_without_turns(self) -> None:
        class EmptyTurnsCase(BaseAgenticCase):
            def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
                del run, report

        with pytest.raises(ValueError, match="at least one turn"):
            EmptyTurnsCase(
                CaseSpec(id="empty_turns", title="Empty turns"),
                EmptyExperimentCase(),
                (),
            )

    def test_run_materializes_invokes_and_validates(self, tmp_path: Path) -> None:
        destination = tmp_path / "case"

        report = CreateRootAgenticCase().run(destination, llm=_create_root_llm())

        report.assert_passed()
        assert report.passed is True
        assert audit_index_path(destination).is_file()

    def test_run_builds_llm_from_explicit_settings(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        settings = LLMSettings()
        llm = _create_root_llm()
        factory = mocker.patch(
            "jarl.agentic.workflow.create_chat_model",
            return_value=llm,
        )

        report = CreateRootAgenticCase().run(
            tmp_path / "settings",
            llm_settings=settings,
        )

        report.assert_passed()
        factory.assert_called()

    def test_run_rejects_llm_and_settings_together(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="either llm or llm_settings"):
            CreateRootAgenticCase().run(
                tmp_path / "invalid",
                llm=_create_root_llm(),
                llm_settings=LLMSettings(),
            )

    def test_run_returns_failed_report_without_pytest_dependency(self, tmp_path: Path) -> None:
        llm = ToolBindingFakeChatModel(messages=iter([AIMessage(content="No changes.")]))

        report = CreateRootAgenticCase().run(tmp_path / "failed", llm=llm)

        assert report.passed is False
        assert report.failures == [
            "Expected the workflow to create one root node.",
            "Expected graph_create_root in the audit index.",
        ]

    def test_run_preserves_multi_turn_state(self, tmp_path: Path) -> None:
        llm = ToolBindingFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="first answer"),
                    AIMessage(content="second answer"),
                ]
            )
        )

        report = TwoTurnAgenticCase().run(tmp_path / "multi", llm=llm)

        report.assert_passed()

    def test_run_rejects_prompt_override_count_mismatch(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Expected 1 prompt override"):
            CreateRootAgenticCase().run(
                tmp_path / "override",
                llm=_create_root_llm(),
                prompts=("one", "two"),
            )

    def test_runs_are_isolated_by_destination(self, tmp_path: Path) -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"

        first_report = CreateRootAgenticCase().run(first_dir, llm=_create_root_llm())
        second_report = CreateRootAgenticCase().run(second_dir, llm=_create_root_llm())

        first_session = load_session(first_dir)
        second_session = load_session(second_dir)
        first_report.assert_passed()
        second_report.assert_passed()
        assert first_session is not None
        assert second_session is not None
        assert first_session.thread_id != second_session.thread_id

    def test_run_propagates_materialization_errors(self, tmp_path: Path) -> None:
        destination = tmp_path / "occupied"
        destination.mkdir()
        (destination / "keep.txt").write_text("keep", encoding="utf-8")

        with pytest.raises(ValueError, match="empty directory"):
            CreateRootAgenticCase().run(destination, llm=_create_root_llm())

    def test_rejects_negative_max_continuations(self) -> None:
        with pytest.raises(ValueError, match="max_continuations"):
            CreateRootAgenticCase(max_continuations=-1)


class TestContinuation:
    def test_builtin_default_matches_constant(self) -> None:
        assert GraphReadWithoutMutationCase().max_continuations == DEFAULT_MAX_CONTINUATIONS

    def test_question_nudges_without_copying_failures(self, tmp_path: Path) -> None:
        llm = RecordingFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="What should I do next?"),
                    AIMessage(content="Still waiting."),
                ]
            )
        )
        destination = tmp_path / "question"

        report = AlwaysFailAgenticCase(max_continuations=1).run(destination, llm=llm)

        assert report.passed is False
        assert report.failures == ["SECRET_FAILURE_TOKEN"]
        assert len(_nudge_batches(llm)) == 1
        assert _batches_mention(llm, "SECRET_FAILURE_TOKEN") is False
        assert "case_continuation" in {event.kind for event in load_progress_events(destination)}

    def test_finish_without_passing_still_nudges_generically(self, tmp_path: Path) -> None:
        llm = RecordingFakeChatModel(
            messages=iter(
                [
                    _tool_call("session_finish", {}, "finish"),
                    AIMessage(content="I declared finish."),
                    AIMessage(content="Continuing after nudge."),
                ]
            )
        )

        report = AlwaysFailAgenticCase(max_continuations=1).run(tmp_path / "early_finish", llm=llm)

        assert report.passed is False
        assert len(_nudge_batches(llm)) == 1
        assert _batches_mention(llm, "SECRET_FAILURE_TOKEN") is False

    def test_finish_and_passed_closes_without_nudge(self, tmp_path: Path) -> None:
        llm = RecordingFakeChatModel(
            messages=iter(
                [
                    _tool_call("graph_create_root", {"label": "baseline"}, "create_root"),
                    AIMessage(content="setup done"),
                    _tool_call("session_finish", {}, "finish"),
                    AIMessage(content="operate done"),
                    AIMessage(content="SHOULD_NOT_BE_REACHED"),
                ]
            )
        )
        destination = tmp_path / "finished"

        report = CreateRootAgenticCase(max_continuations=3).run(destination, llm=llm)

        report.assert_passed()
        assert _nudge_batches(llm) == []
        assert "case_finish" in {event.kind for event in load_progress_events(destination)}
        assert "case_continuation" not in {event.kind for event in load_progress_events(destination)}

    def test_forbidden_mutation_halts_without_further_invokes(self, tmp_path: Path) -> None:
        llm = RecordingFakeChatModel(
            messages=iter(
                [
                    _tool_call(
                        "graph_fork",
                        {"branch": "bad", "label": "x", "prepare": True},
                        "fork_bad",
                    ),
                    AIMessage(content="Forked a branch."),
                    AIMessage(content="SHOULD_NOT_BE_REACHED"),
                ]
            )
        )
        destination = tmp_path / "halt"

        report = GraphReadWithoutMutationCase().run(destination, llm=llm)

        halt_events = [event for event in load_progress_events(destination) if event.kind == "case_halt"]
        assert report.passed is False
        assert report.halted is True
        assert len(llm.captured_message_batches) == 2
        assert halt_events
        assert halt_events[0].data is not None
        assert halt_events[0].data["failures"] == list(report.failures)
        assert report.failures[0] in halt_events[0].message
        assert all(not _batches_mention(llm, failure) for failure in report.failures)
        assert _batches_mention(llm, DEFAULT_NONINTERACTIVE_OBJECTIVE)
        assert "case_continuation" not in {event.kind for event in load_progress_events(destination)}

    def test_cap_stops_after_max_continuations(self, tmp_path: Path) -> None:
        llm = RecordingFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="What next?"),
                    AIMessage(content="Still stuck."),
                    AIMessage(content="Last chance."),
                    AIMessage(content="SHOULD_NOT_BE_REACHED"),
                ]
            )
        )

        report = AlwaysFailAgenticCase(max_continuations=2).run(tmp_path / "cap", llm=llm)

        assert report.passed is False
        assert len(_nudge_batches(llm)) == 2
        assert _batches_mention(llm, "SECRET_FAILURE_TOKEN") is False
        assert len(llm.captured_message_batches) == 3
