"""Tests for versioned checkpoint rollout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import jarl.experiments.io.rollouts as rollouts_io
from jarl.envs.navix.telemetry import (
    NAVIX_EVENT_SLOT_NAMES,
    NavixCaptureProfile,
    NavixRolloutIdentity,
    NavixTraceArrays,
    summarize_navix_trace,
)
from jarl.experiments.io.artifacts import ARTIFACT_KIND_ROLLOUT
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.io.rollouts import (
    ROLLOUT_SCHEMA_VERSION,
    RolloutArtifactManifest,
    compute_rollout_id,
    load_rollout_artifact,
    write_rollout_artifact,
)
from jarl.experiments.node import NodeMetadata, NodeWorkspace

_ACTION_NAMES = (
    "rotate_ccw",
    "rotate_cw",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)


@pytest.fixture()
def workspace(tmp_path: Path) -> NodeWorkspace:
    """Empty node workspace used as rollout artifact owner."""
    return NodeWorkspace(
        node_dir=tmp_path / "node",
        node_metadata=NodeMetadata(id="main_rollout_ab12cd34"),
    )


@pytest.fixture()
def trace() -> NavixTraceArrays:
    """Small padded host trace with one two-step episode."""
    states = np.zeros((1, 4, 4, 4, 3), dtype=np.int32)
    first_person = np.zeros((1, 4, 3, 3, 3), dtype=np.int32)
    positions = np.asarray([[[1, 1], [1, 1], [1, 2], [1, 2]]], dtype=np.int32)
    return NavixTraceArrays(
        full_symbolic=states,
        first_person_symbolic=first_person,
        policy_observations=np.zeros((1, 4, 8), dtype=np.float32),
        player_positions=positions,
        player_directions=np.asarray([[0, 3, 3, 3]], dtype=np.int32),
        player_pockets=-np.ones((1, 4), dtype=np.int32),
        rgb_frames=None,
        actions=np.asarray([[0, 2, -1]], dtype=np.int32),
        rewards=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        step_types=np.asarray([[0, 2, 2]], dtype=np.int32),
        dones=np.asarray([[False, True, True]], dtype=np.bool_),
        active_mask=np.asarray([[True, True, False]], dtype=np.bool_),
        event_happened=np.zeros((1, 3, len(NAVIX_EVENT_SLOT_NAMES)), dtype=np.bool_),
        event_positions=-np.ones((1, 3, len(NAVIX_EVENT_SLOT_NAMES), 2), dtype=np.int32),
        event_colours=-np.ones((1, 3, len(NAVIX_EVENT_SLOT_NAMES)), dtype=np.int32),
        event_types=-np.ones((1, 3, len(NAVIX_EVENT_SLOT_NAMES)), dtype=np.int32),
        mission_present=np.asarray([False], dtype=np.bool_),
        mission_position=-np.ones((1, 2), dtype=np.int32),
        mission_colour=-np.ones((1,), dtype=np.int32),
        mission_event_type=-np.ones((1,), dtype=np.int32),
        episode_returns=np.asarray([1.0], dtype=np.float32),
        episode_lengths=np.asarray([2], dtype=np.int32),
        episode_done=np.asarray([True], dtype=np.bool_),
    )


@pytest.fixture()
def artifact_inputs(
    workspace: NodeWorkspace,
    trace: NavixTraceArrays,
) -> dict[str, object]:
    """Stable writer inputs and matching computed rollout id."""
    source = CheckpointRef(node_id="main_source_cd34ef56", checkpoint_step=64)
    profile = NavixCaptureProfile.analysis()
    config: dict[str, object] = {
        "environment": {"env_id": "Navix-Empty-5x5-v0", "max_episode_steps": 3},
        "algorithm": {"name": "ppo.full_jax.navix"},
    }
    summary = summarize_navix_trace(
        trace,
        identity=NavixRolloutIdentity(
            env_id="Navix-Empty-5x5-v0",
            seed=7,
            node_id=source.node_id,
            checkpoint_step=source.checkpoint_step,
        ),
        action_names=_ACTION_NAMES,
    )
    rollout_id = compute_rollout_id(
        source_checkpoint=source,
        target_node_id=workspace.id,
        algorithm_name="ppo.full_jax.navix",
        env_id="Navix-Empty-5x5-v0",
        seed=7,
        max_steps=3,
        capture_profile=profile,
        effective_config=config,
    )
    return {
        "rollout_id": rollout_id,
        "source_checkpoint": source,
        "algorithm_name": "ppo.full_jax.navix",
        "env_id": "Navix-Empty-5x5-v0",
        "seed": 7,
        "max_steps": 3,
        "capture_profile": profile,
        "effective_config": config,
        "trace": trace,
        "summary": summary,
    }


class TestRolloutIdentity:
    """Rollout ids include every replay-relevant request field."""

    @pytest.mark.parametrize(
        ("field_name", "replacement"),
        [
            pytest.param("seed", 8, id="seed"),
            pytest.param("env_id", "Navix-DoorKey-5x5-v0", id="environment"),
            pytest.param("max_steps", 4, id="horizon"),
            pytest.param(
                "capture_profile",
                NavixCaptureProfile.analysis(record_video=True),
                id="profile",
            ),
            pytest.param(
                "effective_config",
                {"environment": {"env_id": "Navix-Empty-5x5-v0", "seed": 99}},
                id="config",
            ),
            pytest.param("schema_version", ROLLOUT_SCHEMA_VERSION + 1, id="schema"),
        ],
    )
    def test_request_change_changes_rollout_id(
        self,
        workspace: NodeWorkspace,
        artifact_inputs: dict[str, object],
        field_name: str,
        replacement: object,
    ) -> None:
        original = str(artifact_inputs["rollout_id"])
        arguments = {
            key: value
            for key, value in artifact_inputs.items()
            if key
            in {
                "source_checkpoint",
                "algorithm_name",
                "env_id",
                "seed",
                "max_steps",
                "capture_profile",
                "effective_config",
            }
        }
        arguments[field_name] = replacement

        changed = compute_rollout_id(
            target_node_id=workspace.id,
            **arguments,  # type: ignore[arg-type]
        )

        assert changed != original


class TestRolloutArtifactPersistence:
    """Rollout artifacts are compressed, typed, atomic, and idempotent."""

    def test_write_load_register_and_reuse_cache(
        self,
        workspace: NodeWorkspace,
        artifact_inputs: dict[str, object],
    ) -> None:
        first = write_rollout_artifact(
            workspace,
            **artifact_inputs,  # type: ignore[arg-type]
        )
        first_mtime = first.trace_path.stat().st_mtime_ns

        loaded = load_rollout_artifact(first.manifest_path)
        second = write_rollout_artifact(
            workspace,
            **artifact_inputs,  # type: ignore[arg-type]
        )

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert second.trace_path.stat().st_mtime_ns == first_mtime
        assert loaded.trace.actions.shape == (1, 2)
        assert loaded.trace.full_symbolic is not None
        assert loaded.trace.full_symbolic.shape[1] == 3
        assert loaded.manifest.summary.identity.rollout_id == first.ref.rollout_id
        assert loaded.manifest.summary.identity.node_id == workspace.id
        assert loaded.manifest.summary.artifact_paths == (
            f"rollouts/{first.ref.rollout_id}/rollout.json",
            f"rollouts/{first.ref.rollout_id}/trace.npz",
        )
        record = workspace.get_artifact(
            ARTIFACT_KIND_ROLLOUT,
            first.ref.rollout_id,
            64,
        )
        assert record is not None
        assert record.metadata["has_video"] is False

    def test_optional_video_is_stored_from_captured_rgb(
        self,
        workspace: NodeWorkspace,
        artifact_inputs: dict[str, object],
        trace: NavixTraceArrays,
        tmp_path: Path,
    ) -> None:
        profile = NavixCaptureProfile.analysis(record_video=True)
        video_trace = trace._replace(rgb_frames=np.zeros((1, 4, 4, 4, 3), dtype=np.uint8))
        video_path = tmp_path / "source.mp4"
        video_path.write_bytes(b"fake mp4")
        arguments = dict(artifact_inputs)
        arguments["capture_profile"] = profile
        arguments["trace"] = video_trace
        arguments["video_path"] = video_path
        arguments["rollout_id"] = compute_rollout_id(
            source_checkpoint=arguments["source_checkpoint"],  # type: ignore[arg-type]
            target_node_id=workspace.id,
            algorithm_name=str(arguments["algorithm_name"]),
            env_id=str(arguments["env_id"]),
            seed=int(arguments["seed"]),  # type: ignore[arg-type]
            max_steps=int(arguments["max_steps"]),  # type: ignore[arg-type]
            capture_profile=profile,
            effective_config=arguments["effective_config"],  # type: ignore[arg-type]
        )

        result = write_rollout_artifact(
            workspace,
            **arguments,  # type: ignore[arg-type]
        )

        assert result.video_path is not None
        assert result.video_path.read_bytes() == b"fake mp4"
        assert result.manifest.files.video == "rollout.mp4"
        assert result.manifest.summary.artifact_paths[-1].endswith("rollout.mp4")

    def test_failure_before_validation_does_not_register_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: NodeWorkspace,
        artifact_inputs: dict[str, object],
    ) -> None:
        def fail_save(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(rollouts_io.np, "savez_compressed", fail_save)

        with pytest.raises(OSError, match="disk full"):
            write_rollout_artifact(
                workspace,
                **artifact_inputs,  # type: ignore[arg-type]
            )

        assert not workspace.artifacts_registry_path.exists()
        assert not (workspace.rollouts_dir / str(artifact_inputs["rollout_id"])).exists()
        assert list(workspace.rollouts_dir.iterdir()) == []

    def test_manifest_rejects_unknown_schema_version(
        self,
        workspace: NodeWorkspace,
        artifact_inputs: dict[str, object],
    ) -> None:
        result = write_rollout_artifact(
            workspace,
            **artifact_inputs,  # type: ignore[arg-type]
        )
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = ROLLOUT_SCHEMA_VERSION + 1

        with pytest.raises(ValueError, match="Unsupported rollout schema_version"):
            RolloutArtifactManifest.from_dict(payload)

    def test_load_rejects_manifest_identity_tampering(
        self,
        workspace: NodeWorkspace,
        artifact_inputs: dict[str, object],
    ) -> None:
        result = write_rollout_artifact(
            workspace,
            **artifact_inputs,  # type: ignore[arg-type]
        )
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        payload["seed"] = 999
        result.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="identity does not match"):
            load_rollout_artifact(result.manifest_path)
