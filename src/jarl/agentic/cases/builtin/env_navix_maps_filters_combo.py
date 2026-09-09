"""Agentic case that exercises every ``env_navix_maps`` filter together."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._env_navix_maps_case import (
    as_int,
    assert_read_only_catalog_search,
    message_contents,
    navix_maps_events,
    query_mentions,
    request_categories,
    request_truthy,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.demo_tree import DemoTreeCase

__all__ = ["CASE", "EnvNavixMapsFiltersComboCase"]

_LAVA_ENV_IDS = frozenset(
    {
        "Navix-LavaGap-S5-v0",
        "Navix-LavaGap-S6-v0",
        "Navix-LavaGap-S7-v0",
    }
)
"""Canonical LavaGap ids expected from the combined-filter search."""


class EnvNavixMapsFiltersComboCase(BaseAgenticCase):
    """Validate combined category, query, difficulty and transfer filters."""

    def __init__(self) -> None:
        """Initialize the combined-filters search case."""
        super().__init__(
            CaseSpec(
                id="env_navix_maps_filters_combo",
                title="Search maps with all filters",
                description=(
                    "Ask for transfer-ready lava maps in a difficulty band using "
                    "categories, query and difficulty together; no graph mutation."
                ),
                tags=frozenset({"env", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Necesito mapas con lava listos para transfer de checkpoints en JARL, con dificultad entre 40 y 60."
                    "Usa la herramienta con filtros de categoría, palabra clave (query) y dificultad a la vez todas las opciones a la vez."
                    "Dime los ids canónicos. No inventes ids de memoria."
                    "No modifiques el grafo ni entrenes. Si algo falla vuélvelo a intentar."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate that all major catalog filters were exercised."""
        if not assert_read_only_catalog_search(run, report):
            return

        report.check(
            "lava_gap" in request_categories(run),
            "Expected categories to include lava_gap.",
        )
        report.check(
            query_mentions(run, "lava"),
            "Expected query to mention lava.",
        )
        report.check(
            any(as_int(event.request.get("difficulty_min")) is not None for event in navix_maps_events(run)),
            "Expected difficulty_min to be set.",
        )
        report.check(
            any(as_int(event.request.get("difficulty_max")) is not None for event in navix_maps_events(run)),
            "Expected difficulty_max to be set.",
        )
        report.check(
            request_truthy(run, "jarl_transfer_ready_only"),
            "Expected jarl_transfer_ready_only=true.",
        )
        report.check(
            any(env_id in message_contents(run) for env_id in _LAVA_ENV_IDS),
            "Expected the final answer to include at least one LavaGap env id.",
        )


CASE = EnvNavixMapsFiltersComboCase()
"""Built-in combined-filter Navix map search case."""
