"""Unit tests for checkpoint listing in the Runner Lab."""

from __future__ import annotations

from jarl.app.lib.checkpoints_view import CheckpointRow, checkpoint_table


def test_checkpoint_table_formats_aliases() -> None:
    rows = [
        CheckpointRow(
            checkpoint_step=128,
            node_step=128,
            status="saved",
            global_step=128,
            train_return=0.5,
            eval_return=0.4,
            is_latest_alias=True,
            is_final_alias=False,
            is_best_alias=True,
        )
    ]
    frame = checkpoint_table(rows)
    assert frame.iloc[0]["step"] == 128
    assert "latest" in frame.iloc[0]["aliases"]
