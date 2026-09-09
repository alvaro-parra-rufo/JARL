"""Run the built-in agentic case registry with a scripted LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarl.agentic.cases import BaseAgenticCase, get_agentic_case_registry
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel
from tests.helpers.jax_training_subprocess import materialize_case_in_training_subprocess

_CASES = get_agentic_case_registry().values()
_SLOW_CASE_IDS = frozenset(
    {
        "graph_checkpoint_rollout_analysis",
        "graph_checkpoint_rollout_analysis_long",
    }
)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case.spec.id,
            marks=pytest.mark.slow if case.spec.id in _SLOW_CASE_IDS else (),
        )
        for case in _CASES
    ],
)
def test_builtin_agentic_case(
    case: BaseAgenticCase,
    tmp_path: Path,
    agentic_case_llm: ToolBindingFakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case.spec.id in _SLOW_CASE_IDS:
        experiment_case = case.experiment_case
        monkeypatch.setattr(
            experiment_case,
            "materialize",
            lambda destination: materialize_case_in_training_subprocess(
                experiment_case,
                destination,
            ),
        )
    report = case.run(tmp_path / case.spec.id, llm=agentic_case_llm)

    report.assert_passed()
