"""Application services for browsing and executing agentic cases."""

from __future__ import annotations

import json
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from jarl.agentic.cases.base_agentic_case import BaseAgenticCase
from jarl.agentic.cases.catalog import get_agentic_case_registry
from jarl.agentic.cases.labels import agentic_case_status_label
from jarl.agentic.llm import LLMSettings
from jarl.agentic.progress import append_progress_event
from jarl.experiments.cases import CaseRegistry
from jarl.experiments.paths import resolve_cases_root
from jarl.utils.io import write_text_atomic

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

AGENTIC_CASE_RESULT_NAME = "agentic_case_result.json"
"""Filename containing the subprocess-safe agentic case result."""

AGENTIC_CASE_FAVORITES_NAME = ".favorite_runs.json"
"""Filename listing favorite case-run directory names under the cases root."""

AgenticCaseStatus = Literal["passed", "failed", "error"]

__all__ = [
    "AGENTIC_CASE_FAVORITES_NAME",
    "AGENTIC_CASE_RESULT_NAME",
    "AgenticCaseInfo",
    "AgenticCaseResult",
    "AgenticCaseStatus",
    "describe_agentic_case",
    "execute_agentic_case",
    "filter_agentic_case_runs",
    "list_agentic_case_runs",
    "list_agentic_cases",
    "load_agentic_case_result",
    "load_favorite_run_names",
    "resolve_agentic_case_registry",
    "set_favorite_run",
    "write_agentic_case_result",
]


@dataclass(frozen=True, slots=True)
class AgenticCaseInfo:
    """Serializable description of an agentic evaluation case."""

    id: str
    title: str
    description: str
    tags: tuple[str, ...]
    requirements: tuple[str, ...]
    experiment_case_id: str
    turns: tuple[str, ...]
    audit_reads: bool
    launch_options: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgenticCaseResult:
    """Persisted outcome of one agentic case execution."""

    case_id: str
    experiment_dir: Path
    status: AgenticCaseStatus
    failures: tuple[str, ...]
    error: str | None
    started_at: str
    finished_at: str

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        experiment_dir: Path,
    ) -> AgenticCaseResult:
        """Build a result from persisted JSON, anchored to ``experiment_dir``."""
        raw_status = payload["status"]
        status_by_value: dict[object, AgenticCaseStatus] = {
            "passed": "passed",
            "failed": "failed",
            "error": "error",
        }
        status = status_by_value.get(raw_status)
        if status is None:
            msg = f"Unknown agentic case status: {raw_status!r}"
            raise ValueError(msg)
        error = payload.get("error")
        failures = payload.get("failures", ())
        if not isinstance(failures, Sequence) or isinstance(failures, str | bytes):
            msg = f"Invalid agentic case failures: {failures!r}"
            raise TypeError(msg)
        return cls(
            case_id=str(payload["case_id"]),
            experiment_dir=experiment_dir,
            status=status,
            failures=tuple(str(value) for value in failures),
            error=str(error) if error is not None else None,
            started_at=str(payload["started_at"]),
            finished_at=str(payload["finished_at"]),
        )

    @property
    def passed(self) -> bool:
        """Return whether execution and deterministic validation succeeded."""
        return self.status == "passed"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "case_id": self.case_id,
            "experiment_dir": str(self.experiment_dir),
            "status": self.status,
            "passed": self.passed,
            "failures": list(self.failures),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def resolve_agentic_case_registry(
    package: str | None = None,
) -> CaseRegistry[BaseAgenticCase]:
    """Return the built-in or a package-discovered agentic case registry."""
    if package is None:
        return get_agentic_case_registry()
    return CaseRegistry.from_package(package, case_type=BaseAgenticCase)


def list_agentic_cases(
    *,
    package: str | None = None,
) -> tuple[AgenticCaseInfo, ...]:
    """Return stable descriptions for all agentic cases in a catalog."""
    registry = resolve_agentic_case_registry(package)
    return tuple(_agentic_case_info(case) for case in registry)


def list_agentic_case_runs(
    *,
    cases_root: str | Path | None = None,
) -> tuple[AgenticCaseResult, ...]:
    """Return finished case runs under the cases root, newest first.

    Only immediate child directories with a readable `agentic_case_result.json`
    are included. Malformed files are skipped. Each result is anchored to the
    discovered directory, not the path stored in the JSON payload.
    """
    parent = resolve_cases_root(override=cases_root)
    if not parent.is_dir():
        return ()
    results: list[AgenticCaseResult] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        try:
            loaded = load_agentic_case_result(child)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if loaded is None:
            continue
        results.append(loaded)
    results.sort(key=lambda item: (item.finished_at, item.experiment_dir.name), reverse=True)
    return tuple(results)


def filter_agentic_case_runs(
    runs: Sequence[AgenticCaseResult],
    *,
    case_id: str | None = None,
    statuses: Collection[AgenticCaseStatus] | None = None,
    query: str | None = None,
    tags: Collection[str] | None = None,
    tags_by_case_id: Mapping[str, Collection[str]] | None = None,
    favorite_names: Collection[str] | None = None,
    favorites_only: bool = False,
) -> tuple[AgenticCaseResult, ...]:
    """Return runs matching case id, status, tags, query, and favorites.

    Empty ``case_id``, ``statuses``, ``tags``, or ``query`` means no filter on
    that axis. Required tags use ``tags_by_case_id`` (missing ids have no tags).
    ``favorites_only`` keeps directory names listed in ``favorite_names``.
    """
    required_id = (case_id or "").strip()
    status_filter = frozenset(statuses) if statuses else None
    required_tags = frozenset(tag for tag in (tags or ()) if tag)
    tag_map = tags_by_case_id or {}
    allowed_names = frozenset(favorite_names or ()) if favorites_only else None
    needle = (query or "").strip().casefold()
    matched: list[AgenticCaseResult] = []
    for run in runs:
        if required_id and run.case_id != required_id:
            continue
        if status_filter is not None and run.status not in status_filter:
            continue
        run_tags = tuple(tag_map.get(run.case_id, ()))
        if required_tags and not required_tags.issubset(run_tags):
            continue
        if allowed_names is not None and run.experiment_dir.name not in allowed_names:
            continue
        if needle:
            haystack = f"{run.case_id} {run.experiment_dir.name} {' '.join(run_tags)}".casefold()
            if needle not in haystack:
                continue
        matched.append(run)
    return tuple(matched)


def load_favorite_run_names(
    *,
    cases_root: str | Path | None = None,
) -> frozenset[str]:
    """Return favorite run directory names stored under the cases root."""
    path = _favorite_run_names_path(cases_root=cases_root)
    if not path.is_file():
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(name for name in _favorite_names_from_payload(payload) if _is_safe_run_name(name))


def set_favorite_run(
    name: str,
    *,
    favorite: bool,
    cases_root: str | Path | None = None,
) -> frozenset[str]:
    """Add or remove one run directory name from the persisted favorites list."""
    if not _is_safe_run_name(name):
        msg = f"Favorite run name must be a single path component: {name!r}"
        raise ValueError(msg)
    names = set(load_favorite_run_names(cases_root=cases_root))
    if favorite:
        names.add(name)
    else:
        names.discard(name)
    path = _favorite_run_names_path(cases_root=cases_root)
    payload = json.dumps({"names": sorted(names)}, ensure_ascii=False, indent=2)
    write_text_atomic(path, f"{payload}\n")
    return frozenset(names)


def _favorite_run_names_path(*, cases_root: str | Path | None = None) -> Path:
    """Return the favorites JSON path under the cases root."""
    return resolve_cases_root(override=cases_root) / AGENTIC_CASE_FAVORITES_NAME


def _favorite_names_from_payload(payload: object) -> tuple[str, ...]:
    """Extract run directory names from a list or ``{"names": [...]}`` payload."""
    if isinstance(payload, list):
        raw_names = payload
    elif isinstance(payload, dict):
        raw_names = payload.get("names", [])
    else:
        msg = "Favorite runs payload must be a JSON object or list."
        raise TypeError(msg)
    if not isinstance(raw_names, list):
        msg = "Favorite runs names must be a JSON list."
        raise TypeError(msg)
    return tuple(str(item) for item in raw_names)


def _is_safe_run_name(name: str) -> bool:
    """Return whether ``name`` is a single non-empty path component."""
    stripped = name.strip()
    return bool(stripped) and Path(stripped).name == stripped and stripped not in {".", ".."}


def describe_agentic_case(
    case_id: str,
    *,
    package: str | None = None,
) -> AgenticCaseInfo:
    """Return the description of one agentic case."""
    case = resolve_agentic_case_registry(package).get(case_id)
    return _agentic_case_info(case)


def execute_agentic_case(
    case_id: str,
    destination: str | Path,
    *,
    package: str | None = None,
    llm: BaseChatModel | None = None,
    llm_settings: LLMSettings | None = None,
    prompts: Sequence[str] | None = None,
    launch_options: Mapping[str, object] | None = None,
) -> AgenticCaseResult:
    """Execute and persist one agentic case outcome.

    Technical execution errors are represented with status `error`; validation
    failures use status `failed`.
    """
    case = resolve_agentic_case_registry(package).get(case_id)
    experiment_dir = _prepare_destination(destination)
    started_at = _utc_now()
    _emit_case_progress(experiment_dir, f"Iniciando caso {case_id}")
    try:
        report = case.run(
            experiment_dir,
            llm=llm,
            llm_settings=llm_settings,
            prompts=prompts,
            launch_options=launch_options,
        )
        result = AgenticCaseResult(
            case_id=case_id,
            experiment_dir=experiment_dir,
            status="passed" if report.passed else "failed",
            failures=tuple(report.failures),
            error=None,
            started_at=started_at,
            finished_at=_utc_now(),
        )
    except Exception as exc:
        append_progress_event(
            experiment_dir,
            "case_error",
            f"Caso {case_id} error: {type(exc).__name__}: {exc}",
            data={"case_id": case_id},
        )
        _emit_case_progress(experiment_dir, f"Caso {case_id} error: {type(exc).__name__}")
        result = AgenticCaseResult(
            case_id=case_id,
            experiment_dir=experiment_dir,
            status="error",
            failures=(),
            error=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            finished_at=_utc_now(),
        )
    status_message = (
        f"Caso {case_id} pasó"
        if result.status == "passed"
        else f"Caso {case_id}: {agentic_case_status_label(result.status)}"
    )
    if result.failures:
        status_message = f"{status_message}: {'; '.join(result.failures)}"
    append_progress_event(
        experiment_dir,
        "case_end",
        status_message,
        data={
            "case_id": case_id,
            "status": result.status,
            "failures": list(result.failures),
        },
    )
    _emit_case_progress(experiment_dir, status_message)
    write_agentic_case_result(result)
    return result


def write_agentic_case_result(result: AgenticCaseResult) -> Path:
    """Persist an agentic case result atomically in its experiment directory."""
    path = result.experiment_dir / AGENTIC_CASE_RESULT_NAME
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    return write_text_atomic(path, f"{payload}\n")


def load_agentic_case_result(
    experiment_dir: str | Path,
) -> AgenticCaseResult | None:
    """Load a persisted result, returning `None` while no result exists.

    ``experiment_dir`` on the returned object is the directory argument, not
    the path snapshot stored in the JSON payload.
    """
    resolved = Path(experiment_dir).expanduser().resolve()
    path = resolved / AGENTIC_CASE_RESULT_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Agentic case result must be a JSON object: {path}"
        raise TypeError(msg)
    return AgenticCaseResult.from_payload(payload, experiment_dir=resolved)


def _agentic_case_info(case: BaseAgenticCase) -> AgenticCaseInfo:
    spec = case.spec
    return AgenticCaseInfo(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        tags=tuple(sorted(spec.tags)),
        requirements=tuple(sorted(spec.requirements)),
        experiment_case_id=case.experiment_case.spec.id,
        turns=tuple(turn.prompt for turn in case.turns),
        audit_reads=case.audit_reads,
        launch_options=tuple(option.to_dict() for option in case.launch_options),
    )


def _prepare_destination(destination: str | Path) -> Path:
    experiment_dir = Path(destination).expanduser().resolve()
    if experiment_dir.exists():
        if not experiment_dir.is_dir() or any(experiment_dir.iterdir()):
            msg = f"Agentic case destination must be an empty directory: {experiment_dir}"
            raise ValueError(msg)
    else:
        experiment_dir.mkdir(parents=True)
    return experiment_dir


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit_case_progress(experiment_dir: Path, message: str) -> None:
    """Mirror progress events to subprocess stdout for the case log tail."""
    del experiment_dir
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()
