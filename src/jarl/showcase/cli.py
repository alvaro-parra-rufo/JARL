"""CLI for the JARL showcase pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from jarl.experiments.paths import resolve_workspace_root
from jarl.showcase.config import DEFAULT_SHOWCASE_WANDB_PROJECT, build_plan, validate_plan
from jarl.showcase.executor import ShowcaseExecutor
from jarl.utils.env import load_project_env

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the showcase pipeline."""
    load_project_env()
    parser = argparse.ArgumentParser(description="Generate a full JARL showcase experiment tree.")
    parser.add_argument("--name", default="jarl-showcase", help="Experiment directory name.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Experiment parent directory. Default: JARL_EXPERIMENTS_ROOT or results/dags.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "ppo_gru"),
        default=None,
        help="Trainer architecture (overrides scenario YAML; default: scenario or ppo).",
    )
    parser.add_argument("--scale", choices=("smoke", "medium", "large"), default="medium")
    parser.add_argument("--scenario", default=None, help="Scenario YAML path or name.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without training.")
    parser.add_argument("--reuse", action="store_true", help="Resume; skip completed nodes.")
    parser.add_argument("--from-node", default=None, help="Resume from logical node name.")
    parser.add_argument("--force", action="store_true", help="Allow reruns on existing workspace.")
    parser.add_argument("--wandb-online", action="store_true", help="Force W&B online mode.")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help=f"W&B project name (default: {DEFAULT_SHOWCASE_WANDB_PROJECT}).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    workspace_root = resolve_workspace_root(override=args.workspace)
    plan = build_plan(
        experiment_name=args.name,
        workspace_root=workspace_root,
        scenario_path=args.scenario,
        scale_name=args.scale,
        algorithm=args.algorithm,
        wandb_online=args.wandb_online,
        wandb_project=args.wandb_project or DEFAULT_SHOWCASE_WANDB_PROJECT,
    )
    errors = validate_plan(plan)
    if errors:
        for error in errors:
            sys.stderr.write(f"validation error: {error}\n")
        return 1

    executor = ShowcaseExecutor(
        plan,
        reuse=args.reuse,
        from_node=args.from_node,
        force=args.force,
    )
    try:
        return executor.run(dry_run=args.dry_run)
    except FileExistsError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 3
