"""Agentic case for filtering Navix maps by difficulty."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._env_navix_maps_case import (
    as_int,
    assert_read_only_catalog_search,
    difficulty_bounds_used,
    message_contents,
    navix_maps_events,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.demo_tree import DemoTreeCase

__all__ = ["CASE", "EnvNavixMapsDifficultyEasyCase"]

_EASY_ENV_IDS = frozenset(
    {
        "Navix-Empty-5x5-v0",
        "Navix-Empty-6x6-v0",
        "Navix-Empty-Random-5x5-v0",
        "Navix-Empty-8x8-v0",
        "Navix-DoorKey-5x5-v0",
    }
)
"""Easy-ish env ids expected when asking for low difficulty maps."""

_MAX_EASY_DIFFICULTY = 20
"""Upper difficulty bound the agent should request for this case."""


class EnvNavixMapsDifficultyEasyCase(BaseAgenticCase):
    """Validate difficulty-range filtering for easy Navix maps."""

    def __init__(self) -> None:
        """Initialize the easy-difficulty search case."""
        super().__init__(
            CaseSpec(
                id="env_navix_maps_difficulty_easy",
                title="Search easy maps by difficulty",
                description=(
                    "Ask for low-difficulty maps so the agent uses "
                    "difficulty_min/max on env_navix_maps; no graph mutation."
                ),
                tags=frozenset({"env", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Quiero mapas fáciles, con dificultad como máximo 20 "
                    "en la escala 1-100 del catálogo. Busca con la herramienta y "
                    "dime ids canónicos. No inventes ids de memoria. "
                    "No modifiques el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate difficulty filter evidence and unchanged graph shape."""
        if not assert_read_only_catalog_search(run, report):
            return
        report.check(
            difficulty_bounds_used(run),
            "Expected env_navix_maps to set difficulty_min and/or difficulty_max.",
        )
        report.check(
            any(
                (bound := as_int(event.request.get("difficulty_max"))) is not None and bound <= _MAX_EASY_DIFFICULTY
                for event in navix_maps_events(run)
            ),
            f"Expected difficulty_max <= {_MAX_EASY_DIFFICULTY}.",
        )
        report.check(
            any(env_id in message_contents(run) for env_id in _EASY_ENV_IDS),
            "Expected the final answer to include at least one easy env id.",
        )


CASE = EnvNavixMapsDifficultyEasyCase()
"""Built-in easy-difficulty Navix map search case."""
