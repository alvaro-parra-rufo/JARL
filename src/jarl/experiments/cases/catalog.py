"""Lazy access to the built-in experiment case catalog."""

from __future__ import annotations

from functools import cache
from typing import cast

from jarl.experiments.cases.case import ExperimentCase
from jarl.experiments.cases.registry import CaseRegistry
from jarl.training.config import RLRunConfig

__all__ = ["get_experiment_case_registry"]

BUILTIN_EXPERIMENT_CASES_PACKAGE = "jarl.experiments.cases.builtin"
"""Import package containing built-in experiment cases."""


@cache
def get_experiment_case_registry() -> CaseRegistry[ExperimentCase[RLRunConfig]]:
    """Discover and cache the built-in experiment case catalog on first use."""
    registry = CaseRegistry.from_package(
        BUILTIN_EXPERIMENT_CASES_PACKAGE,
        case_type=ExperimentCase,
    )
    return cast("CaseRegistry[ExperimentCase[RLRunConfig]]", registry)
