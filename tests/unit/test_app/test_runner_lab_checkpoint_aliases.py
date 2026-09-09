"""Tests for checkpoint alias helpers."""

from __future__ import annotations

from pytest_mock import MockerFixture

from jarl.app.lib.checkpoints_view import format_checkpoint_option
from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_LATEST,
    checkpoint_ref_from_alias,
)


class TestCheckpointAliases:
    def test_format_checkpoint_option_with_alias(self) -> None:
        from jarl.app.lib.checkpoints_view import CheckpointRow

        checkpoint_row = CheckpointRow(
            checkpoint_step=128,
            node_step=128,
            status="saved",
            global_step=128,
            train_return=0.5,
            eval_return=0.6,
            is_latest_alias=True,
            is_final_alias=False,
            is_best_alias=False,
        )
        label = format_checkpoint_option(checkpoint_row)
        assert "step 128" in label
        assert "latest" in label

    def test_checkpoint_ref_from_alias_none_when_missing(self, mocker: MockerFixture) -> None:
        workspace = mocker.Mock()
        workspace.id = "node_a"
        workspace.resolve_checkpoint_alias.return_value = None
        assert checkpoint_ref_from_alias(workspace, CHECKPOINT_ALIAS_LATEST) is None

    def test_checkpoint_ref_from_alias_resolves(self, mocker: MockerFixture) -> None:
        workspace = mocker.Mock()
        workspace.id = "node_a"
        record = mocker.Mock()
        record.checkpoint_step = 64
        workspace.resolve_checkpoint_alias.return_value = record
        ref = checkpoint_ref_from_alias(workspace, CHECKPOINT_ALIAS_LATEST)
        assert ref is not None
        assert ref.checkpoint_step == 64
        assert ref.node_id == "node_a"
