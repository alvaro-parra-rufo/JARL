"""Command-line interface for reusable experiment cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jarl.experiments.cases.services import (
    ExperimentCaseInfo,
    MaterializedExperimentCase,
    allocate_case_destination,
    describe_experiment_case,
    list_experiment_cases,
    materialize_experiment_case,
)
from jarl.experiments.paths import resolve_cases_root
from jarl.utils.env import load_project_env

CASE_EXIT_SUCCESS = 0
"""Exit code for successful case commands."""

CASE_EXIT_INVALID = 2
"""Exit code for invalid catalog selections or destinations."""

CASE_EXIT_ERROR = 3
"""Exit code for unexpected materialization errors."""

__all__ = [
    "CASE_EXIT_ERROR",
    "CASE_EXIT_INVALID",
    "CASE_EXIT_SUCCESS",
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment-case command parser."""
    parser = argparse.ArgumentParser(description="Browse and materialize reusable JARL experiment cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available experiment cases.")
    _add_catalog_arguments(list_parser)

    describe_parser = subparsers.add_parser("describe", help="Describe one experiment case.")
    describe_parser.add_argument("case_id")
    _add_catalog_arguments(describe_parser)

    materialize_parser = subparsers.add_parser("materialize", help="Materialize one experiment case.")
    materialize_parser.add_argument("case_id")
    materialize_parser.add_argument("--destination", type=Path, default=None)
    materialize_parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Parent directory for case runs (default: JARL_CASES_ROOT or results/cases).",
    )
    materialize_parser.add_argument("--run-name", default=None)
    _add_catalog_arguments(materialize_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an experiment-case command."""
    load_project_env()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            cases = list_experiment_cases(package=args.package)
            _write_case_list(cases, as_json=args.json)
            return CASE_EXIT_SUCCESS
        if args.command == "describe":
            case = describe_experiment_case(args.case_id, package=args.package)
            _write_case(case, as_json=args.json)
            return CASE_EXIT_SUCCESS
        describe_experiment_case(args.case_id, package=args.package)
        destination = _resolve_destination(args)
        result = materialize_experiment_case(
            args.case_id,
            destination,
            package=args.package,
        )
        _write_materialized(result, as_json=args.json)
        return CASE_EXIT_SUCCESS
    except (FileExistsError, ImportError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return CASE_EXIT_INVALID
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return CASE_EXIT_ERROR


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package", default=None, help="Importable package containing explicit CASE exports.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _resolve_destination(args: argparse.Namespace) -> Path:
    if args.destination is not None:
        if args.run_name is not None:
            raise ValueError("--run-name cannot be combined with --destination.")
        return args.destination
    root = resolve_cases_root(override=args.workspace)
    return allocate_case_destination(args.case_id, cases_root=root, run_name=args.run_name)


def _write_case_list(
    cases: tuple[ExperimentCaseInfo, ...],
    *,
    as_json: bool,
) -> None:
    if as_json:
        _write_json({"cases": [case.to_dict() for case in cases]})
        return
    for case in cases:
        tags = ", ".join(case.tags) or "—"
        sys.stdout.write(f"{case.id}\t{case.title}\t{tags}\n")


def _write_case(case: ExperimentCaseInfo, *, as_json: bool) -> None:
    if as_json:
        _write_json({"case": case.to_dict()})
        return
    sys.stdout.write(f"{case.title} ({case.id})\n")
    if case.description:
        sys.stdout.write(f"{case.description}\n")
    sys.stdout.write(f"Tags: {', '.join(case.tags) or '—'}\n")
    sys.stdout.write(f"Requirements: {', '.join(case.requirements) or '—'}\n")
    sys.stdout.write(f"Config: {case.config_type}\n")


def _write_materialized(
    result: MaterializedExperimentCase,
    *,
    as_json: bool,
) -> None:
    if as_json:
        _write_json({"result": result.to_dict()})
        return
    sys.stdout.write(f"Materialized {result.case_id} at {result.experiment_dir}\n")
    for alias, node_id in sorted(result.aliases.items()):
        sys.stdout.write(f"{alias}: {node_id}\n")


def _write_json(payload: dict[str, object]) -> None:
    sys.stdout.write(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n")
