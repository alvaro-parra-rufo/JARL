"""Lazy access to the built-in agentic case catalog."""

from __future__ import annotations

from functools import cache

from jarl.agentic.cases.base_agentic_case import BaseAgenticCase
from jarl.experiments.cases import CaseRegistry

__all__ = ["get_agentic_case_registry"]

BUILTIN_AGENTIC_CASES_PACKAGE = "jarl.agentic.cases.builtin"
"""Import package containing built-in agentic cases."""


@cache
def get_agentic_case_registry() -> CaseRegistry[BaseAgenticCase]:
    """Discover and cache the built-in agentic case catalog on first use."""
    return CaseRegistry.from_package(
        BUILTIN_AGENTIC_CASES_PACKAGE,
        case_type=BaseAgenticCase,
    )
