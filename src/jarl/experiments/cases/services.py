"""Application services for browsing and materializing experiment cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from jarl.experiments.cases.case import ExperimentCase
from jarl.experiments.cases.catalog import get_experiment_case_registry
from jarl.experiments.cases.registry import CaseRegistry
from jarl.experiments.paths import resolve_cases_root
from jarl.training.config import RLRunConfig
from jarl.utils.ids import short_uuid

__all__ = [
    "ExperimentCaseInfo",
    "MaterializedExperimentCase",
    "allocate_case_destination",
    "describe_experiment_case",
    "list_experiment_cases",
    "materialize_experiment_case",
    "resolve_experiment_case_registry",
]


@dataclass(frozen=True, slots=True)
class ExperimentCaseInfo:
    """Serializable description of a reusable experiment case."""

    id: str
    title: str
    description: str
    tags: tuple[str, ...]
    requirements: tuple[str, ...]
    config_type: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MaterializedExperimentCase:
    """Result of materializing one experiment case."""

    case_id: str
    experiment_dir: Path
    aliases: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "case_id": self.case_id,
            "experiment_dir": str(self.experiment_dir),
            "aliases": dict(self.aliases),
        }


def resolve_experiment_case_registry(
    package: str | None = None,
) -> CaseRegistry[ExperimentCase[RLRunConfig]]:
    """Return the built-in or a package-discovered experiment case registry."""
    if package is None:
        return get_experiment_case_registry()
    return CaseRegistry.from_package(package, case_type=ExperimentCase)


def list_experiment_cases(
    *,
    package: str | None = None,
) -> tuple[ExperimentCaseInfo, ...]:
    """Return stable descriptions for all experiment cases in a catalog."""
    registry = resolve_experiment_case_registry(package)
    return tuple(_experiment_case_info(case) for case in registry)


def describe_experiment_case(
    case_id: str,
    *,
    package: str | None = None,
) -> ExperimentCaseInfo:
    """Return the description of one experiment case."""
    case = resolve_experiment_case_registry(package).get(case_id)
    return _experiment_case_info(case)


def allocate_case_destination(
    case_id: str,
    *,
    cases_root: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    """Reserve an empty top-level experiment directory for a case run.

    Args:
        case_id: Stable case identifier included in generated names.
        cases_root: Parent directory for case runs. Defaults to ``resolve_cases_root()``.
        run_name: Optional explicit directory name.

    Returns:
        Newly created empty directory.

    Raises:
        FileExistsError: If the selected directory already exists.
        ValueError: If `run_name` is not a single safe path component.
    """
    name = run_name or _default_run_name(case_id)
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Case run name must be a single path component.")
    parent = resolve_cases_root(override=cases_root)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / name
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def materialize_experiment_case(
    case_id: str,
    destination: str | Path,
    *,
    package: str | None = None,
) -> MaterializedExperimentCase:
    """Materialize one registered experiment case in an isolated directory."""
    case = resolve_experiment_case_registry(package).get(case_id)
    context = case.materialize(destination)
    return MaterializedExperimentCase(
        case_id=case_id,
        experiment_dir=context.experiment_dir,
        aliases=dict(context.aliases),
    )


def _experiment_case_info(
    case: ExperimentCase[RLRunConfig],
) -> ExperimentCaseInfo:
    spec = case.spec
    return ExperimentCaseInfo(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        tags=tuple(sorted(spec.tags)),
        requirements=tuple(sorted(spec.requirements)),
        config_type=case.config_cls.__name__,
    )


def _default_run_name(case_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{case_id}-{timestamp}-{short_uuid()}"
