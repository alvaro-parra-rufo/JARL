"""Run the built-in agentic case registry with a configured real LLM."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jarl.agentic.cases import BaseAgenticCase, get_agentic_case_registry

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

pytestmark = pytest.mark.llm

_CASES = get_agentic_case_registry().values()
_SLOW_CASE_IDS = frozenset({"navix_empty_variant_spec"})


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
    agentic_case_llm: BaseChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case.spec.id == "navix_empty_variant_spec":
        monkeypatch.setattr(case, "require_trained_eval", True)
    report = case.run(tmp_path / case.spec.id, llm=agentic_case_llm)

    report.assert_passed()
