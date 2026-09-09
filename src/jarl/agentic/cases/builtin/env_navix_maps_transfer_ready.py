"""Agentic case for listing JARL transfer-ready Navix maps."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import AgenticCaseRun, AgenticTurn, BaseAgenticCase
from jarl.agentic.cases.builtin._env_navix_maps_case import (
    assert_read_only_catalog_search,
    message_contents,
    request_truthy,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.demo_tree import DemoTreeCase

__all__ = ["CASE", "EnvNavixMapsTransferReadyCase"]

_TRANSFER_READY_ENV_IDS = frozenset(
    {
        "Navix-Empty-5x5-v0",
        "Navix-DoorKey-5x5-v0",
        "Navix-FourRooms-v0",
        "Navix-LavaGap-S5-v0",
    }
)
"""Sample transfer-ready ids the agent may surface."""


class EnvNavixMapsTransferReadyCase(BaseAgenticCase):
    """Validate ``jarl_transfer_ready_only`` filtering."""

    def __init__(self) -> None:
        """Initialize the transfer-ready search case."""
        super().__init__(
            CaseSpec(
                id="env_navix_maps_transfer_ready",
                title="Search JARL transfer-ready maps",
                description=(
                    "Ask for checkpoint-transfer-ready maps so the agent sets "
                    "jarl_transfer_ready_only; no graph mutation."
                ),
                tags=frozenset({"env", "operate", "read"}),
                requirements=frozenset({"llm"}),
            ),
            DemoTreeCase(),
            (
                AgenticTurn(
                    "Lista solo los mapas listos para transferir checkpoints "
                    "en JARL. Usa el filtro de la herramienta y dime ids canónicos. "
                    "No inventes ids de memoria. No modifiques el grafo ni entrenes."
                ),
            ),
        )

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate transfer-ready filter evidence and unchanged graph shape."""
        if not assert_read_only_catalog_search(run, report):
            return
        report.check(
            request_truthy(run, "jarl_transfer_ready_only"),
            "Expected jarl_transfer_ready_only=true.",
        )
        report.check(
            any(env_id in message_contents(run) for env_id in _TRANSFER_READY_ENV_IDS),
            "Expected the final answer to include at least one transfer-ready env id.",
        )


CASE = EnvNavixMapsTransferReadyCase()
"""Built-in transfer-ready Navix map search case."""
