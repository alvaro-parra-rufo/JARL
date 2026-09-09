"""Tests for checkpoint metadata registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.io.checkpoints import (
    CHECKPOINT_ALIAS_BEST,
    CHECKPOINT_ALIAS_FINAL,
    CHECKPOINT_ALIAS_LATEST,
    CheckpointOrigin,
    CheckpointRecord,
    CheckpointRef,
    CheckpointRegistry,
    CheckpointStatus,
)


class TestCheckpointRecord:
    """Tests for checkpoint record serialization."""

    @pytest.mark.parametrize(
        ("global_step", "optimizer_updates"),
        [
            pytest.param(None, None, id="optional_none"),
            pytest.param(100, 50, id="optional_set"),
        ],
    )
    def test_round_trip_preserves_fields(
        self,
        global_step: int | None,
        optimizer_updates: int | None,
    ) -> None:
        record = CheckpointRecord.for_step(
            node_step=5,
            checkpoint_step=5,
            metrics={"loss": 0.1},
            global_step=global_step,
            optimizer_updates=optimizer_updates,
            origin=CheckpointOrigin(node_id="parent", checkpoint_step=3),
        )

        restored = CheckpointRecord.from_dict(record.to_dict())

        assert restored.node_step == 5
        assert restored.checkpoint_step == 5
        assert restored.status == CheckpointStatus.SAVED
        assert restored.metrics == {"loss": 0.1}
        assert restored.global_step == global_step
        assert restored.optimizer_updates == optimizer_updates
        assert restored.origin == CheckpointOrigin(node_id="parent", checkpoint_step=3)

    def test_checkpoint_ref_round_trip(self) -> None:
        ref = CheckpointRef(node_id="main_baseline_ab12cd34", checkpoint_step=10)

        restored = CheckpointRef.from_dict(ref.to_dict())

        assert restored == ref


class TestCheckpointRegistry:
    """Tests for checkpoints.json persistence."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "checkpoints.json"
        registry = CheckpointRegistry(registry_path)
        record = CheckpointRecord.for_step(node_step=1, checkpoint_step=1, metrics={"loss": 0.5})
        registry.register(record)
        registry.set_alias_latest(1)
        registry.set_alias_final(1)
        registry.promote_best(1, metric_name="loss", metric_value=0.5, reason="manual")
        registry.pin(1)
        registry.save()

        restored = CheckpointRegistry(registry_path)
        restored.load()

        assert len(restored.list()) == 1
        assert restored.resolve_alias(CHECKPOINT_ALIAS_LATEST) == record
        assert restored.resolve_alias(CHECKPOINT_ALIAS_FINAL) == record
        assert restored.resolve_alias(CHECKPOINT_ALIAS_BEST) == record
        assert restored.pinned_checkpoint_steps == frozenset({1})

    def test_written_payload_shape(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "checkpoints.json"
        registry = CheckpointRegistry(registry_path)
        registry.register(CheckpointRecord.for_step(node_step=2, checkpoint_step=2))
        registry.save()

        payload = json.loads(registry_path.read_text(encoding="utf-8"))

        assert "checkpoints" in payload
        assert "aliases" in payload
        assert "pinned_checkpoint_steps" in payload

    def test_load_rejects_phantom_alias(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "checkpoints.json"
        registry_path.write_text(
            json.dumps(
                {
                    "checkpoints": [],
                    "aliases": {"latest": 5},
                    "pinned_checkpoint_steps": [],
                }
            ),
            encoding="utf-8",
        )

        registry = CheckpointRegistry(registry_path)

        with pytest.raises(ValueError, match="points to missing step"):
            registry.load()
