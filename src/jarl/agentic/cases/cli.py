"""Command-line interface for agentic evaluation cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from jarl.agentic.cases.services import (
    AgenticCaseInfo,
    AgenticCaseResult,
    describe_agentic_case,
    execute_agentic_case,
    list_agentic_cases,
)
from jarl.agentic.llm import ENV_LLM_CONFIG, resolve_llm_catalog_ref
from jarl.experiments.cases import allocate_case_destination
from jarl.experiments.paths import resolve_cases_root
from jarl.utils.env import load_project_env

AGENTIC_CASE_EXIT_PASSED = 0
"""Exit code for a case that passes deterministic validation."""

AGENTIC_CASE_EXIT_FAILED = 1
"""Exit code for a completed case with validation failures."""

AGENTIC_CASE_EXIT_INVALID = 2
"""Exit code for invalid catalog selections or destinations."""

AGENTIC_CASE_EXIT_ERROR = 3
"""Exit code for LLM, workflow, or setup failures."""

__all__ = [
    "AGENTIC_CASE_EXIT_ERROR",
    "AGENTIC_CASE_EXIT_FAILED",
    "AGENTIC_CASE_EXIT_INVALID",
    "AGENTIC_CASE_EXIT_PASSED",
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the agentic-case command parser."""
    parser = argparse.ArgumentParser(description="Browse and execute reusable JARL agentic cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available agentic cases.")
    _add_catalog_arguments(list_parser)

    describe_parser = subparsers.add_parser("describe", help="Describe one agentic case.")
    describe_parser.add_argument("case_id")
    _add_catalog_arguments(describe_parser)

    run_parser = subparsers.add_parser("run", help="Execute one agentic case.")
    run_parser.add_argument("case_id")
    run_parser.add_argument("--destination", type=Path, default=None)
    run_parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Parent directory for case runs (default: JARL_CASES_ROOT or results/cases).",
    )
    run_parser.add_argument("--run-name", default=None)
    run_parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Override one turn prompt. Repeat once per turn, in order.",
    )
    run_parser.add_argument(
        "--llm-config",
        default=None,
        help=(
            "LLM catalog YAML path or packaged stem (custom/default). Uses catalog assignments for main and subagents."
        ),
    )
    run_parser.add_argument(
        "--launch-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Force one declared case launch option (repeatable).",
    )
    _add_catalog_arguments(run_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an agentic-case command."""
    load_project_env()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            cases = list_agentic_cases(package=args.package)
            _write_case_list(cases, as_json=args.json)
            return AGENTIC_CASE_EXIT_PASSED
        if args.command == "describe":
            case = describe_agentic_case(args.case_id, package=args.package)
            _write_case(case, as_json=args.json)
            return AGENTIC_CASE_EXIT_PASSED

        describe_agentic_case(args.case_id, package=args.package)
        destination = _resolve_destination(args)
        _apply_llm_config(args.llm_config)
        result = execute_agentic_case(
            args.case_id,
            destination,
            package=args.package,
            prompts=args.prompt,
            launch_options=_parse_launch_options(args.launch_option),
        )
        _write_result(result, as_json=args.json)
        return _result_exit_code(result)
    except (FileExistsError, ImportError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return AGENTIC_CASE_EXIT_INVALID
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return AGENTIC_CASE_EXIT_ERROR


def _apply_llm_config(llm_config: str | None) -> None:
    """Point ``JARL_LLM_CONFIG`` at the selected catalog YAML, if any."""
    if llm_config is None:
        return
    os.environ[ENV_LLM_CONFIG] = str(resolve_llm_catalog_ref(llm_config))


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package", default=None, help="Importable package containing explicit CASE exports.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _parse_launch_options(entries: Sequence[str] | None) -> dict[str, object] | None:
    """Parse ``KEY=VALUE`` launch options from the CLI."""
    if not entries:
        return None
    parsed: dict[str, object] = {}
    for entry in entries:
        if "=" not in entry:
            msg = f"Launch option must use KEY=VALUE format, got {entry!r}."
            raise ValueError(msg)
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Launch option key must not be empty in {entry!r}.")
        parsed[key] = value.strip()
    return parsed


def _resolve_destination(args: argparse.Namespace) -> Path:
    if args.destination is not None:
        if args.run_name is not None:
            raise ValueError("--run-name cannot be combined with --destination.")
        return args.destination
    root = resolve_cases_root(override=args.workspace)
    return allocate_case_destination(args.case_id, cases_root=root, run_name=args.run_name)


def _result_exit_code(result: AgenticCaseResult) -> int:
    if result.status == "passed":
        return AGENTIC_CASE_EXIT_PASSED
    if result.status == "failed":
        return AGENTIC_CASE_EXIT_FAILED
    return AGENTIC_CASE_EXIT_ERROR


def _write_case_list(
    cases: tuple[AgenticCaseInfo, ...],
    *,
    as_json: bool,
) -> None:
    if as_json:
        _write_json({"cases": [case.to_dict() for case in cases]})
        return
    for case in cases:
        tags = ", ".join(case.tags) or "—"
        sys.stdout.write(f"{case.id}\t{case.title}\t{tags}\n")


def _write_case(case: AgenticCaseInfo, *, as_json: bool) -> None:
    if as_json:
        _write_json({"case": case.to_dict()})
        return
    sys.stdout.write(f"{case.title} ({case.id})\n")
    if case.description:
        sys.stdout.write(f"{case.description}\n")
    sys.stdout.write(f"Scenario: {case.experiment_case_id}\n")
    sys.stdout.write(f"Tags: {', '.join(case.tags) or '—'}\n")
    sys.stdout.write(f"Requirements: {', '.join(case.requirements) or '—'}\n")
    for index, turn in enumerate(case.turns, start=1):
        sys.stdout.write(f"Turn {index}: {turn}\n")


def _write_result(result: AgenticCaseResult, *, as_json: bool) -> None:
    if as_json:
        _write_json({"result": result.to_dict()})
        return
    sys.stdout.write(f"{result.case_id}: {result.status.upper()}\n")
    sys.stdout.write(f"Experiment: {result.experiment_dir}\n")
    for failure in result.failures:
        sys.stdout.write(f"- {failure}\n")
    if result.error is not None:
        sys.stdout.write(f"Error: {result.error}\n")


def _write_json(payload: dict[str, object]) -> None:
    sys.stdout.write(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n")
