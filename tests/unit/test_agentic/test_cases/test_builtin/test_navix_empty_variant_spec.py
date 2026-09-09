"""Anti-leak and catalog checks for the EmptyVariant agentic case."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage
from pytest_mock import MockerFixture

from jarl.agentic.cases.builtin._graph_checkpoint_rollout_analysis import RECORD_VIDEO_LAUNCH_OPTION
from jarl.agentic.cases.builtin._navix_empty_variant_eval import original_reward_payload
from jarl.agentic.cases.builtin.navix_empty_variant_spec import CASE, NavixEmptyVariantSpecCase
from jarl.agentic.cases.validation import ValidationReport
from jarl.agentic.prompts import CONTINUATION_SYSTEM_PROMPT
from tests.helpers.fake_chat_model import RecordingFakeChatModel

_LEAKED_SURFACE = (
    "dopamina",
    "dopamine",
    "bonus",
    "trap",
    "trampa",
    "cell_entry",
    "hidden_cell",
    "floor_cell",
    "occupancy",
    "compatible_env_ids",
    "scenario_reward",
    "25.6",
    "16.6",
    "15.93",
    "r_farm",
    "decenas",
    "g=4",
    "goal_reached=26",
)


class TestNavixEmptyVariantSpecCase:
    """CaseSpec and user turns must stay on the task, not overlay oracles."""

    def test_catalog_identity(self) -> None:
        assert CASE.spec.id == "navix_empty_variant_spec"
        assert isinstance(CASE, NavixEmptyVariantSpecCase)
        assert CASE.spec.objective == ""
        assert CASE.require_trained_eval is False
        assert CASE.launch_options == (RECORD_VIDEO_LAUNCH_OPTION,)
        payload = original_reward_payload()
        assert payload["goal_reached"] == 1.0
        assert payload["goal_approach"] == 0.0

    def test_prompt_and_spec_omit_overlay_and_thresholds(self, mocker: MockerFixture) -> None:
        surfaces = " ".join(
            [
                CASE.spec.title,
                CASE.spec.description,
                CASE.spec.objective,
                *(turn.prompt for turn in CASE.turns),
                CASE.continuation_prompt(
                    mocker.Mock(),
                    ValidationReport(case_id=CASE.spec.id),
                    finished=True,
                ),
            ]
        ).casefold()

        assert "26" not in surfaces
        assert "recompensa" not in surfaces
        assert "reward mix" not in surfaces
        for token in _LEAKED_SURFACE:
            assert token not in surfaces

    def test_premature_finish_nudge_does_not_copy_failures(self, mocker: MockerFixture) -> None:
        text = CASE.continuation_prompt(
            mocker.Mock(),
            ValidationReport(
                case_id=CASE.spec.id,
                failures=["Expected graph_set_reward to change the configurable mix."],
            ),
            finished=True,
        )

        assert "El objetivo no fue alcanzado" in text
        assert "graph_set_reward" not in text
        assert (
            CASE.continuation_prompt(
                mocker.Mock(),
                ValidationReport(case_id=CASE.spec.id),
                finished=False,
            )
            == CONTINUATION_SYSTEM_PROMPT
        )

    def test_finish_without_solving_sends_objective_nudge(self, tmp_path: Path) -> None:
        llm = RecordingFakeChatModel(
            messages=iter(
                (
                    _tool_call("session_finish", {}, "finish"),
                    AIMessage(content="He terminado."),
                    AIMessage(content="Sigo."),
                )
            )
        )
        report = NavixEmptyVariantSpecCase(max_continuations=1).run(
            tmp_path / "early_finish",
            llm=llm,
        )

        assert not report.passed
        assert any(
            isinstance(message, SystemMessage) and "El objetivo no fue alcanzado" in str(message.content)
            for batch in llm.captured_message_batches
            for message in batch
        )
        joined = "\n".join(str(message.content) for batch in llm.captured_message_batches for message in batch)
        assert "Expected graph_set_reward" not in joined
        assert "occupancy" not in joined.casefold()


def _tool_call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
    """Build one scripted AI tool-call message."""
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
