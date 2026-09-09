"""Tests for tool attachment discovery."""

from __future__ import annotations

from pathlib import Path

from jarl.agentic.tool_attachments import extract_tool_attachments, resolve_attachment_path
from jarl.experiments.io.layout import ExperimentLayout


class TestToolAttachments:
    def test_extracts_rollout_video_from_artifact_paths(self) -> None:
        payload = {
            "identity": {"node_id": "main_root_ab12cd34"},
            "artifact_paths": [
                "rollouts/abc/rollout.json",
                "rollouts/abc/trace.npz",
                "rollouts/abc/rollout.mp4",
            ],
        }

        attachments = extract_tool_attachments("graph_checkpoint_rollout", payload)

        assert len(attachments) == 1
        assert attachments[0].kind == "video"
        assert attachments[0].relative_path.endswith("rollout.mp4")

    def test_resolve_attachment_path_uses_node_workspace(self, tmp_path: Path) -> None:
        node_id = "main_root_ab12cd34"
        layout = ExperimentLayout(tmp_path)
        node_dir = layout.node_dir(node_id)
        node_dir.mkdir(parents=True)
        relative = "rollouts/demo/rollout.mp4"
        video_path = node_dir / relative
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"mp4")

        resolved = resolve_attachment_path(
            tmp_path,
            {"identity": {"node_id": node_id}},
            relative,
        )

        assert resolved == video_path
