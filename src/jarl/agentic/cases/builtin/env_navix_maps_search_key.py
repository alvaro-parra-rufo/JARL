"""Agentic case for searching all Navix maps that involve a key."""

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

__all__ = ["CASE", "EnvNavixMapsSearchKeyCase"]

_KEY_CATEGORIES = frozenset({"door_key", "key_corridor"})
"""Category keys that cover Navix maps with a key."""

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
"""Canonical DoorKey ids expected in a key-family search answer."""

_KEY_CORRIDOR_ENV_IDS = frozenset(
    {
        "Navix-KeyCorridorS3R1-v0",
        "Navix-KeyCorridorS3R2-v0",
        "Navix-KeyCorridorS3R3-v0",
        "Navix-KeyCorridorS4R3-v0",
        "Navix-KeyCorridorS5R3-v0",
        "Navix-KeyCorridorS6R3-v0",
    }
)
"""Canonical KeyCorridor ids expected in a key-family search answer."""


class EnvNavixMapsSearchKeyCase(BaseAgenticCase):
    """Validate a multi-category catalog search for maps that use a key."""

    def __init__(self) -> None:
        """Initialize the key-maps search case."""
        super().__init__(
            CaseSpec(
                id="env_navix_maps_search_key",
                title="Search maps that use a key",
                description=(
                    "Ask for every map that uses a key so the agent must cover "
                    "door_key and key_corridor via env_navix_maps; no graph mutation."
                ),
                tags=frozenset({"env", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Quiero todos los mapas que usen llave. "
                    "Busca en el catálogo todas las familias relevantes y dime los "
                    "ids canónicos. No inventes ids de memoria: consulta la herramienta. "
                    "No modifiques el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate multi-category key search evidence and unchanged graph shape."""
        if not assert_read_only_catalog_search(run, report):
            return
        report.check(
            request_categories(run) >= _KEY_CATEGORIES,
            "Expected env_navix_maps categories to cover door_key and key_corridor.",
        )
        final_text = message_contents(run)
        report.check(
            any(env_id in final_text for env_id in _DOOR_KEY_ENV_IDS),
            "Expected the final answer to include at least one DoorKey env id.",
        )
        report.check(
            any(env_id in final_text for env_id in _KEY_CORRIDOR_ENV_IDS),
            "Expected the final answer to include at least one KeyCorridor env id.",
        )


CASE = EnvNavixMapsSearchKeyCase()
"""Built-in multi-category Navix key-map search case."""
