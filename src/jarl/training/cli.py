"""CLI argument parsing for ``python -m jarl.training.run``."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig
from jarl.training.launch import clear_train_worker_pid
from jarl.training.registry import import_trainer_callable, resolve_trainer_from_name
from jarl.training.run_overrides import retrain_overrides_from_config
from jarl.training.runner import resume_training, run_training
from jarl.training.trainer import Trainer

__all__ = [
    "build_config_overrides",
    "load_config_snapshot",
    "main",
    "parse_args",
    "resolve_trainer",
    "resolve_trainer_for_node",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse runner CLI arguments.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Run jarl RL training on an experiment node.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment root directory.")
    parser.add_argument("--node-id", default=None, help="Target node id (defaults to current node).")
    parser.add_argument("--create-root", action="store_true", help="Create a root node when empty.")
    parser.add_argument("--config-json", default=None, help="Full ``RLRunConfig`` snapshot JSON path.")
    parser.add_argument("--resume", action="store_true", help="Resume a failed or interrupted node.")
    parser.add_argument("--label", default="", help="Optional label when creating a root node.")
    parser.add_argument(
        "--trainer",
        default=None,
        help="Optional trainer override as ``module:callable``. Defaults to ``algorithm.name``.",
    )
    parser.add_argument("--env-id", default=None, help="Override ``environment.env_id``.")
    parser.add_argument(
        "--algorithm-name",
        default=None,
        help="Override ``algorithm.name`` (registry trainer resolution).",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override ``algorithm.total_timesteps``.")
    parser.add_argument("--nr-envs", type=int, default=None, help="Override ``environment.nr_envs``.")
    parser.add_argument("--nr-steps", type=int, default=None, help="Override ``algorithm.nr_steps``.")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Override ``algorithm.minibatch_size``.",
    )
    parser.add_argument(
        "--obs-encoding-dim",
        type=int,
        default=None,
        help="Override ``algorithm.obs_encoding_dim``.",
    )
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=None,
        help="Override ``algorithm.gru_hidden_dim``.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="Override ``algorithm.evaluation_and_save_frequency``.",
    )
    parser.add_argument("--no-eval", action="store_true", help="Disable ``algorithm.evaluation_active``.")
    parser.add_argument("--save-model", action="store_true", help="Enable ``runner.save_model``.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable ``tracking.track_wandb``.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable ``tracking.track_tensorboard``.")
    return parser.parse_args(list(argv) if argv is not None else None)


def build_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into nested config overrides.

    Args:
        args: Parsed CLI namespace from ``parse_args``.

    Returns:
        Override dictionary suitable for ``RLRunConfig.apply_overrides``.
    """
    overrides: dict[str, Any] = {}
    for arg_name, config_path in _CLI_SCALAR_OVERRIDES:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[config_path] = value
    if args.no_eval:
        overrides["algorithm.evaluation_active"] = False
    if args.save_model:
        overrides["runner.save_model"] = True
    if args.no_wandb:
        overrides["tracking.track_wandb"] = False
    if args.no_tensorboard:
        overrides["tracking.track_tensorboard"] = False
    return overrides


_CLI_SCALAR_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("env_id", "environment.env_id"),
    ("algorithm_name", "algorithm.name"),
    ("total_timesteps", "algorithm.total_timesteps"),
    ("nr_envs", "environment.nr_envs"),
    ("nr_steps", "algorithm.nr_steps"),
    ("minibatch_size", "algorithm.minibatch_size"),
    ("obs_encoding_dim", "algorithm.obs_encoding_dim"),
    ("gru_hidden_dim", "algorithm.gru_hidden_dim"),
    ("eval_frequency", "algorithm.evaluation_and_save_frequency"),
)


def load_config_snapshot(path: Path) -> RLRunConfig:
    """Load and validate an ``RLRunConfig`` JSON snapshot from disk."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RLRunConfig.model_validate(raw)


def resolve_trainer(args: argparse.Namespace, config: RLRunConfig | None) -> Trainer:
    """Resolve the trainer callable from CLI flags and config.

    Args:
        args: Parsed CLI namespace from ``parse_args``.
        config: Resolved or base config used to read ``algorithm.name``.

    Returns:
        Trainer callable.

    Raises:
        SystemExit: If no trainer can be resolved.
    """
    if args.trainer is not None:
        return import_trainer_callable(args.trainer)
    if config is not None:
        return resolve_trainer_from_name(config.algorithm.name)
    raise SystemExit("Provide --trainer or run against a config with a registered algorithm.name.")


def resolve_trainer_for_node(
    exp_dir: Path,
    *,
    node_id: str | None,
    fallback_config: RLRunConfig,
) -> Trainer:
    """Resolve the trainer from the target node lineage when training an existing node."""
    if node_id is None:
        return resolve_trainer_from_name(fallback_config.algorithm.name)
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    node_config = graph.resolve_config(graph.get_node(node_id))
    return resolve_trainer_from_name(node_config.algorithm.name)


def _resolve_cli_config(args: argparse.Namespace) -> RLRunConfig | None:
    """Return config used to resolve the trainer for flag-based CLI invocations."""
    if args.create_root:
        return RLRunConfig()
    exp_dir = Path(args.experiment_dir)
    graph = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    workspace = graph.get_node(args.node_id) if args.node_id is not None else graph.current_node
    return graph.resolve_config(workspace)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint delegating to ``run_training`` or ``resume_training``."""
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    _install_sigterm_interrupt()
    args = parse_args(argv)
    exp_dir = Path(args.experiment_dir)
    try:
        return _run_parsed(args, exp_dir)
    finally:
        clear_train_worker_pid(exp_dir)


def _run_parsed(args: argparse.Namespace, exp_dir: Path) -> int:
    """Dispatch training after CLI arguments are parsed."""
    if args.config_json is not None:
        config = load_config_snapshot(Path(args.config_json))
        trainer = resolve_trainer_for_node(exp_dir, node_id=args.node_id, fallback_config=config)
        overrides = retrain_overrides_from_config(config)
        if args.resume:
            if args.node_id is None:
                msg = "--node-id is required for resume."
                raise SystemExit(msg)
            resume_training(
                trainer=trainer,
                experiment_dir=exp_dir,
                node=args.node_id,
                config_overrides=overrides,
            )
            return 0

        run_training(
            trainer=trainer,
            experiment_dir=exp_dir,
            config=config if args.create_root else None,
            config_overrides=None if args.create_root else overrides,
            create_root=args.create_root,
            node=args.node_id,
            label=args.label,
        )
        return 0

    config = _resolve_cli_config(args) if not args.create_root else RLRunConfig()
    trainer = resolve_trainer(args, config)
    base_config = RLRunConfig() if args.create_root else None
    run_training(
        trainer=trainer,
        experiment_dir=args.experiment_dir,
        node=args.node_id,
        config=base_config,
        config_overrides=build_config_overrides(args),
        create_root=args.create_root,
        label=args.label,
    )
    return 0


def _install_sigterm_interrupt() -> None:
    """Map SIGTERM to KeyboardInterrupt so the node exits as ``interrupted``."""

    def _handler(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    sys.exit(main())
