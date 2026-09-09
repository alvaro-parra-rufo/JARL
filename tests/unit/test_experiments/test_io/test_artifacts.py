"""Tests for artifact records and registry persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.experiments.io.artifacts import (
    ARTIFACT_KIND_ROLLOUT,
    ARTIFACT_KIND_VIDEO,
    ARTIFACT_KIND_VIDEO_METRICS,
    ArtifactRecord,
    ArtifactRegistry,
)
from jarl.experiments.io.layout import VIDEO_METRICS_JSONL_FILENAME
from jarl.experiments.node import NodeMetadata, NodeWorkspace


class TestArtifactRecord:
    """Tests for the ArtifactRecord contract."""

    @pytest.mark.parametrize(
        ("name", "kind", "relative_path", "step", "metadata"),
        [
            pytest.param("policy", "video", "videos/policy.mp4", 100, {"fps": 30}, id="video"),
            pytest.param("notes", "generic", "exports/notes.txt", 0, {}, id="generic"),
        ],
    )
    def test_round_trip_dict(
        self,
        name: str,
        kind: str,
        relative_path: str,
        step: int,
        metadata: dict[str, object],
    ) -> None:
        record = ArtifactRecord(
            name=name,
            kind=kind,
            relative_path=relative_path,
            step=step,
            metadata=metadata,
        )

        restored = ArtifactRecord.from_dict(record.to_dict())

        assert restored == record
        assert restored.identity == (kind, name, step)

    def test_video_factory_uses_videos_dir(self) -> None:
        record = ArtifactRecord.video("rollout", step=42, metadata={"seed": 1})

        assert record.kind == ARTIFACT_KIND_VIDEO
        assert record.relative_path == "videos/rollout.mp4"
        assert record.metadata == {"seed": 1}

    def test_video_metrics_factory_points_to_jsonl(self) -> None:
        record = ArtifactRecord.video_metrics(step=10)

        assert record.kind == ARTIFACT_KIND_VIDEO_METRICS
        assert record.relative_path == VIDEO_METRICS_JSONL_FILENAME
        assert record.name == VIDEO_METRICS_JSONL_FILENAME

    def test_rollout_factory_points_to_versioned_manifest(self) -> None:
        record = ArtifactRecord.rollout(
            "0123456789abcdef01234567",
            step=42,
            metadata={"seed": 7},
        )

        assert record.kind == ARTIFACT_KIND_ROLLOUT
        assert record.relative_path == ("rollouts/0123456789abcdef01234567/rollout.json")
        assert record.metadata == {"seed": 7}


class TestArtifactRegistry:
    """Tests for registry persistence and lookup."""

    def test_round_trip_from_disk(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "artifacts.json"
        root = tmp_path / "node"
        root.mkdir()
        (root / "videos").mkdir()
        video_path = root / "videos" / "policy.mp4"
        video_path.write_text("fake", encoding="utf-8")

        original = ArtifactRegistry(registry_path, root=root)
        original.register(ArtifactRecord.video("policy", step=5))
        original.register(ArtifactRecord.video_metrics(step=5, metadata={"count": 1}))
        original.register(
            ArtifactRecord(
                name="summary",
                kind="generic",
                relative_path="exports/summary.txt",
                step=5,
            )
        )
        original.save()

        restored = ArtifactRegistry(registry_path, root=root)
        restored.load()

        assert restored.list() == original.list()
        assert restored.resolve_path(ARTIFACT_KIND_VIDEO, "policy", 5) == video_path
        assert restored.resolve_path(ARTIFACT_KIND_VIDEO_METRICS, VIDEO_METRICS_JSONL_FILENAME, 5) == (
            root / VIDEO_METRICS_JSONL_FILENAME
        )
        assert restored.get("generic", "summary", 5) is not None

    def test_load_rejects_phantom_model_alias(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "artifacts.json"
        root = tmp_path / "node"
        root.mkdir()
        registry_path.write_text(
            json.dumps(
                {
                    "artifacts": [],
                    "model_aliases": {"latest": {"name": "weights", "step": 3}},
                }
            ),
            encoding="utf-8",
        )

        registry = ArtifactRegistry(registry_path, root=root)

        with pytest.raises(ValueError, match="points to missing artifact"):
            registry.load()

    def test_register_replaces_same_identity(self, tmp_path: Path) -> None:
        registry = ArtifactRegistry(tmp_path / "artifacts.json", root=tmp_path)

        registry.register(ArtifactRecord(name="policy", kind=ARTIFACT_KIND_VIDEO, relative_path="videos/a.mp4", step=1))
        registry.register(
            ArtifactRecord(
                name="policy",
                kind=ARTIFACT_KIND_VIDEO,
                relative_path="videos/b.mp4",
                step=1,
                metadata={"updated": True},
            )
        )

        record = registry.get(ARTIFACT_KIND_VIDEO, "policy", 1)

        assert record is not None
        assert record.relative_path == "videos/b.mp4"
        assert record.metadata == {"updated": True}
        assert len(registry.list()) == 1

    def test_register_rejects_path_escape(self, tmp_path: Path) -> None:
        registry = ArtifactRegistry(tmp_path / "artifacts.json", root=tmp_path)

        with pytest.raises(ValueError, match="escapes the node root"):
            registry.register(ArtifactRecord(name="bad", kind="generic", relative_path="../outside.txt", step=0))


class TestNodeWorkspaceArtifacts:
    """Tests for artifact APIs on NodeWorkspace."""

    @pytest.fixture()
    def workspace(self, tmp_path: Path) -> NodeWorkspace:
        """Empty node workspace."""
        return NodeWorkspace(
            node_dir=tmp_path / "node",
            node_metadata=NodeMetadata(id="main_test_ab12cd34"),
        )

    def test_register_and_resolve_video_video_metrics_and_generic(
        self,
        workspace: NodeWorkspace,
    ) -> None:
        workspace.videos_dir.mkdir(parents=True, exist_ok=True)
        video_path = workspace.videos_dir / "policy.mp4"
        video_path.write_text("fake", encoding="utf-8")
        workspace.video_metrics_jsonl_path.write_text('{"step": 1}\n', encoding="utf-8")
        exports_dir = workspace.artifacts_registry_path.parent / "exports"
        exports_dir.mkdir()
        notes_path = exports_dir / "notes.txt"
        notes_path.write_text("hello", encoding="utf-8")

        workspace.register_artifact(ArtifactRecord.video("policy", step=100))
        workspace.register_artifact(ArtifactRecord.video_metrics(step=100))
        workspace.register_artifact(
            ArtifactRecord(name="notes", kind="generic", relative_path="exports/notes.txt", step=100)
        )

        assert workspace.resolve_artifact_path(ARTIFACT_KIND_VIDEO, "policy", 100) == video_path
        assert (
            workspace.resolve_artifact_path(
                ARTIFACT_KIND_VIDEO_METRICS,
                VIDEO_METRICS_JSONL_FILENAME,
                100,
            )
            == workspace.video_metrics_jsonl_path
        )
        assert workspace.resolve_artifact_path("generic", "notes", 100) == notes_path
        assert len(workspace.list_artifacts()) == 3

        payload = json.loads(workspace.artifacts_registry_path.read_text(encoding="utf-8"))
        assert len(payload["artifacts"]) == 3
        assert payload["artifacts"][0]["kind"] in {ARTIFACT_KIND_VIDEO, ARTIFACT_KIND_VIDEO_METRICS, "generic"}

    def test_registry_persists_across_workspace_instances(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "node"
        metadata = NodeMetadata(id="main_test_ab12cd34")

        first = NodeWorkspace(node_dir=node_dir, node_metadata=metadata)
        first.register_artifact(ArtifactRecord.video("rollout", step=7))

        second = NodeWorkspace(node_dir=node_dir, node_metadata=metadata)

        assert second.get_artifact(ARTIFACT_KIND_VIDEO, "rollout", 7) is not None

    def test_resolve_missing_artifact_raises_key_error(self, workspace: NodeWorkspace) -> None:
        with pytest.raises(KeyError, match="No artifact registered"):
            workspace.resolve_artifact_path("generic", "missing", 0)
