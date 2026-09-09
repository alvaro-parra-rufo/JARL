"""Agentic case for keyword search over the Navix catalog."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._env_navix_maps_case import (
    assert_read_only_catalog_search,
    message_contents,
    query_mentions,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.demo_tree import DemoTreeCase

__all__ = ["CASE", "EnvNavixMapsQueryLavaCase"]

_LAVA_ENV_IDS = frozenset(
    {
        "Navix-LavaGap-S5-v0",
        "Navix-LavaGap-S6-v0",
        "Navix-LavaGap-S7-v0",
    }
)
"""Canonical LavaGap ids expected after a lava keyword search."""


class EnvNavixMapsQueryLavaCase(BaseAgenticCase):
    """Validate keyword ``query`` search for lava maps."""

    def __init__(self) -> None:
        """Initialize the lava keyword search case."""
        super().__init__(
            CaseSpec(
                id="env_navix_maps_query_lava",
                title="Search maps by lava keyword",
                description=(
                    "Ask for lava-related maps via natural language so the agent "
                    "uses env_navix_maps query; no graph mutation."
                ),
                tags=frozenset({"env", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Busca en el catálogo los mapas relacionados con lava "
                    "usando búsqueda por palabra clave. Dime los ids canónicos. "
                    "No inventes ids de memoria: consulta la herramienta. "
                    "No modifiques el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate keyword search evidence and unchanged graph shape."""
        if not assert_read_only_catalog_search(run, report):
            return
        report.check(
            query_mentions(run, "lava"),
            "Expected env_navix_maps query to mention lava.",
        )
        report.check(
            any(env_id in message_contents(run) for env_id in _LAVA_ENV_IDS),
            "Expected the final answer to include at least one LavaGap env id.",
        )


CASE = EnvNavixMapsQueryLavaCase()
"""Built-in Navix lava keyword search case."""
