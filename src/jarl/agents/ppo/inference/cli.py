"""CLI argument parsing for ``python -m jarl.agents.ppo.inference``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from jarl.agents.ppo.inference.render import render_checkpoint_greedy_video

__all__ = ["main", "parse_args"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse greedy checkpoint video CLI arguments."""
    parser = argparse.ArgumentParser(description="Render greedy checkpoint videos for a node.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--role", default="source_checkpoint")
    parser.add_argument("--name-prefix", default="checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--final-video-episodes", type=int, default=None)
    parser.add_argument("--load-from-node-id", default=None)
    parser.add_argument("--use-parent-checkpoint", action="store_true")
    parser.add_argument("--parent-id", default=None)
    parser.add_argument("--child-id", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging() -> None:
    """Send inference subprocess logs to stdout."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Render greedy checkpoint videos and return a process exit code."""
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    args = parse_args(argv)
    _configure_logging()
    logger = logging.getLogger(__name__)
    try:
        encoded_paths = render_checkpoint_greedy_video(
            experiment_dir=Path(args.experiment_dir),
            node_id=args.node_id,
            checkpoint_step=int(args.checkpoint_step),
            env_id=args.env_id,
            role=args.role,
            name_prefix=args.name_prefix,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            final_video_episodes=args.final_video_episodes,
            load_from_node_id=args.load_from_node_id,
            use_parent_checkpoint=args.use_parent_checkpoint,
            parent_id=args.parent_id,
            child_id=args.child_id,
        )
    except Exception:
        logger.exception(
            "Inference failed for node=%s checkpoint=%s prefix=%s",
            args.node_id,
            args.checkpoint_step,
            args.name_prefix,
        )
        return 1

    logger.info(
        "Inference completed: node=%s checkpoint=%s prefix=%s videos=%s",
        args.node_id,
        args.checkpoint_step,
        args.name_prefix,
        ", ".join(str(path) for path in encoded_paths) or "none",
    )
    return 0
