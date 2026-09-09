"""Agentic case for discovering Navix maps via the catalog tool."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._env_navix_maps_case import (
    assert_read_only_catalog_search,
    message_contents,
    request_categories,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.demo_tree import DemoTreeCase

__all__ = ["CASE", "EnvNavixMapsReadCase"]

_DOOR_KEY_ENV_IDS = frozenset(
    {
        "Navix-DoorKey-5x5-v0",
        "Navix-DoorKey-6x6-v0",
        "Navix-DoorKey-8x8-v0",
        "Navix-DoorKey-16x16-v0",
        "Navix-DoorKey-5x5-Random-v0",
        "Navix-DoorKey-6x6-Random-v0",
        "Navix-DoorKey-8x8-Random-v0",
        "Navix-DoorKey-16x16-Random-v0",
    }
)
"""Canonical DoorKey ids the agent should surface after a catalog search."""


class EnvNavixMapsReadCase(BaseAgenticCase):
    """Validate that the agent searches the Navix catalog instead of guessing ids."""

    def __init__(self) -> None:
        """Initialize the Navix map search case."""
        super().__init__(
            CaseSpec(
                id="env_navix_maps_read",
                title="Search maps by family",
                description=(
                    "Ask for DoorKey-family maps in natural language so the agent "
                    "must call env_navix_maps; no graph mutation."
                ),
                tags=frozenset({"env", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Necesito saber qué mapas hay de la familia puerta y llave. "
                    "Busca en el catálogo y dime los ids canónicos. "
                    "No inventes ids de memoria: consulta la herramienta. "
                    "No modifiques el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate catalog search evidence and unchanged graph shape."""
        if not assert_read_only_catalog_search(run, report):
            return
        report.check(
            "door_key" in request_categories(run),
            "Expected env_navix_maps to search with categories including door_key.",
        )
        report.check(
            any(env_id in message_contents(run) for env_id in _DOOR_KEY_ENV_IDS),
            "Expected the final answer to include at least one canonical DoorKey env id.",
        )


CASE = EnvNavixMapsReadCase()
"""Built-in Navix map search case."""
