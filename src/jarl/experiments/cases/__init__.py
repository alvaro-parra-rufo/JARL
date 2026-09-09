"""Reusable experiment case definitions, discovery, and materialization."""

from jarl.experiments.cases.case import (
    CaseDefinition,
    ExperimentCase,
    ExperimentCaseContext,
)
from jarl.experiments.cases.catalog import get_experiment_case_registry
from jarl.experiments.cases.metadata import ExperimentCaseMetadata
from jarl.experiments.cases.registry import CaseRegistry
from jarl.experiments.cases.services import (
    ExperimentCaseInfo,
    MaterializedExperimentCase,
    allocate_case_destination,
    describe_experiment_case,
    list_experiment_cases,
    materialize_experiment_case,
    resolve_experiment_case_registry,
)
from jarl.experiments.cases.spec import CaseSpec

__all__ = [
    "CaseDefinition",
    "CaseRegistry",
    "CaseSpec",
    "ExperimentCase",
    "ExperimentCaseContext",
    "ExperimentCaseInfo",
    "ExperimentCaseMetadata",
    "MaterializedExperimentCase",
    "allocate_case_destination",
    "describe_experiment_case",
    "get_experiment_case_registry",
    "list_experiment_cases",
    "materialize_experiment_case",
    "resolve_experiment_case_registry",
]
