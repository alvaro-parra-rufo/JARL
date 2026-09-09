"""Tests for ``jarl.agents.ppo.inference.cli``."""

from __future__ import annotations

from pathlib import Path

from jarl.agents.ppo.inference.cli import parse_args


class TestInferenceCliArgs:
    """Greedy video CLI flag parsing."""

    def test_parse_args_required_fields(self, tmp_path: Path) -> None:
        args = parse_args(
            [
                "--experiment-dir",
                str(tmp_path / "exp"),
                "--node-id",
                "node_a",
                "--checkpoint-step",
                "128",
                "--env-id",
                "Navix-Empty-5x5-v0",
            ]
        )

        assert args.node_id == "node_a"
        assert args.checkpoint_step == 128
        assert args.env_id == "Navix-Empty-5x5-v0"
        assert args.name_prefix == "checkpoint"
