"""Tests for analyzed checkpoint rollout operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from jarl.envs.navix.catalog import NavixMapContract, register_map
from jarl.envs.navix.telemetry import (
    NAVIX_EVENT_SLOT_NAMES,
    NavixRolloutIdentity,
    NavixTraceArrays,
    summarize_navix_trace,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.inference.run import CheckpointInferenceResult
from jarl.operations.contracts.constants import (
    CHECKPOINT_ROLLOUT_MAX_STEPS,
    CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT,
    CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT,
)
from jarl.operations.graph.checkpoint_rollout import (
    CheckpointRolloutRequest,
    checkpoint_rollout,
)
from jarl.training.config import RLRunConfig
from tests.helpers.minimal_navix_configs import (
    MINIMAL_NAVIX_ENV_ID,
    minimal_ppo_run_config,
)

_CHECKPOINT_STEP = 4
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
def rollout_graph(tmp_path: Path) -> ExperimentGraph[RLRunConfig]:
    """Prepared Navix graph with one registered checkpoint."""
    config = minimal_ppo_run_config()
    graph = ExperimentGraph(tmp_path / "exp", base_config=config)
    workspace = graph.create_root(
        config,
        branch="main",
        label="baseline",
        prepare=True,
    )
    workspace.init_checkpoint_manager(
        max_to_keep=None,
        save_interval_steps=1,
    )
    workspace.save_checkpoint(
        _CHECKPOINT_STEP,
        {"value": np.asarray([1.0], dtype=np.float32)},
        force=True,
    )
    graph.save()
    return graph


class TestCheckpointRolloutDefaults:
    """Optional video fields fall back to operation-level defaults."""

    def test_omitted_video_fields_use_operation_defaults(self) -> None:
        request = CheckpointRolloutRequest(checkpoint_step=0, seed=0)

        assert request.record_video is CHECKPOINT_ROLLOUT_RECORD_VIDEO_DEFAULT
        assert request.video_view_mode == CHECKPOINT_ROLLOUT_VIDEO_VIEW_MODE_DEFAULT


class TestCheckpointRolloutValidation:
    """Requests fail before inference for invalid or unavailable inputs."""

    @pytest.mark.parametrize(
        ("rollout_request", "message"),
        [
            pytest.param(
                CheckpointRolloutRequest(checkpoint_step=-1, seed=0),
                "checkpoint_step",
                id="checkpoint",
            ),
            pytest.param(
                CheckpointRolloutRequest(
                    checkpoint_step=0,
                    seed=0,
                    max_steps=0,
                ),
                "max_steps",
                id="zero_horizon",
            ),
            pytest.param(
                CheckpointRolloutRequest(
                    checkpoint_step=0,
                    seed=0,
                    max_steps=CHECKPOINT_ROLLOUT_MAX_STEPS + 1,
                ),
                "max_steps",
                id="horizon_cap",
            ),
        ],
    )
    def test_rejects_invalid_request(
        self,
        rollout_graph: ExperimentGraph[RLRunConfig],
        rollout_request: CheckpointRolloutRequest,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            checkpoint_rollout(rollout_graph, rollout_request)

    def test_rejects_missing_exact_checkpoint(
        self,
        rollout_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        with pytest.raises(ValueError, match="step 999"):
            checkpoint_rollout(
                rollout_graph,
                CheckpointRolloutRequest(
                    checkpoint_step=999,
                    seed=0,
                    max_steps=3,
                ),
            )

    def test_rejects_incompatible_environment(
        self,
        rollout_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        env_id = "Navix-Incompatible-Rollout-Test-v0"
        register_map(
            NavixMapContract(
                env_id=env_id,
                processed_obs_shape=(147,),
                action_names=_ACTION_NAMES,
                transfer_group="incompatible_test",
            )
        )

        with pytest.raises(ValueError, match="transfer groups"):
            checkpoint_rollout(
                rollout_graph,
                CheckpointRolloutRequest(
                    checkpoint_step=_CHECKPOINT_STEP,
                    seed=0,
                    env_id=env_id,
                    max_steps=3,
                ),
            )


class TestCheckpointRolloutExecution:
    """The operation composes inference, persistence, cache, and video."""

    def test_persists_summary_and_reuses_exact_cache(
        self,
        mocker: MockerFixture,
        rollout_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = rollout_graph.current_node
        inference = mocker.patch(
            "jarl.operations.graph.checkpoint_rollout.run_checkpoint_inference",
            return_value=_inference_result(
                node_id=workspace.id,
                env_id=MINIMAL_NAVIX_ENV_ID,
                seed=7,
            ),
        )
        setup = mocker.patch(
            "jarl.operations.graph.checkpoint_rollout.setup_jax",
        )
        request = CheckpointRolloutRequest(
            checkpoint_step=_CHECKPOINT_STEP,
            seed=7,
            max_steps=3,
        )

        first = checkpoint_rollout(rollout_graph, request)
        second = checkpoint_rollout(rollout_graph, request)

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert second.rollout_id == first.rollout_id
        assert first.summary.artifact_paths == (
            f"rollouts/{first.rollout_id}/rollout.json",
            f"rollouts/{first.rollout_id}/trace.npz",
        )
        assert (workspace.path / first.summary.artifact_paths[0]).is_file()
        assert (workspace.path / first.summary.artifact_paths[1]).is_file()
        inference.assert_called_once()
        setup.assert_called_once()
        compact = second.to_compact_dict()
        assert compact["cache_hit"] is True
        assert compact["identity"]["seed"] == 7  # type: ignore[index]
        assert "trace" not in compact
        assert "rgb_frames" not in str(compact)

    def test_applies_seed_and_compatible_target_environment(
        self,
        mocker: MockerFixture,
        rollout_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = rollout_graph.current_node
        target_env_id = "Navix-DoorKey-5x5-v0"
        inference = mocker.patch(
            "jarl.operations.graph.checkpoint_rollout.run_checkpoint_inference",
            return_value=_inference_result(
                node_id=workspace.id,
                env_id=target_env_id,
                seed=23,
            ),
        )
        mocker.patch("jarl.operations.graph.checkpoint_rollout.setup_jax")

        response = checkpoint_rollout(
            rollout_graph,
            CheckpointRolloutRequest(
                checkpoint_step=_CHECKPOINT_STEP,
                seed=23,
                env_id=target_env_id,
                max_steps=3,
            ),
        )

        target_config = inference.call_args.args[1]
        capture_request = inference.call_args.args[2]
        assert target_config.environment.env_id == target_env_id
        assert target_config.environment.seed == 23
        assert target_config.environment.max_episode_steps == 3
        assert capture_request.seeds == (23,)
        assert capture_request.max_steps == 3
        assert capture_request.summary_limits.max_events == 64
        assert capture_request.summary_limits.max_path_positions == 128
        assert response.env_id == target_env_id

    def test_video_uses_captured_frames_without_second_inference(
        self,
        mocker: MockerFixture,
        rollout_graph: ExperimentGraph[RLRunConfig],
    ) -> None:
        workspace = rollout_graph.current_node
        inference = mocker.patch(
            "jarl.operations.graph.checkpoint_rollout.run_checkpoint_inference",
            return_value=_inference_result(
                node_id=workspace.id,
                env_id=MINIMAL_NAVIX_ENV_ID,
                seed=5,
                record_video=True,
            ),
        )
        mocker.patch("jarl.operations.graph.checkpoint_rollout.setup_jax")

        def encode(
            frames: np.ndarray,
            destination: Path,
            *,
            fps: int,
            scale: int,
        ) -> Path:
            assert frames.shape == (3, 4, 4, 3)
            assert fps > 0
            assert scale > 0
            destination.write_bytes(b"mp4")
            return destination

        encoder = mocker.patch(
            "jarl.operations.graph.checkpoint_rollout.encode_rgb_video",
            side_effect=encode,
        )

        response = checkpoint_rollout(
            rollout_graph,
            CheckpointRolloutRequest(
                checkpoint_step=_CHECKPOINT_STEP,
                seed=5,
                max_steps=3,
                record_video=True,
                video_view_mode="first_person",
            ),
        )

        capture_request = inference.call_args.args[2]
        assert capture_request.capture_profile.capture_symbolic is True
        assert capture_request.capture_profile.rgb_view_mode == "first_person"
        inference.assert_called_once()
        encoder.assert_called_once()
        video_relative = response.summary.artifact_paths[-1]
        assert video_relative.endswith("rollout.mp4")
        assert (workspace.path / video_relative).read_bytes() == b"mp4"


def _inference_result(
    *,
    node_id: str,
    env_id: str,
    seed: int,
    record_video: bool = False,
) -> CheckpointInferenceResult:
    trace = _trace(record_video=record_video)
    summary = summarize_navix_trace(
        trace,
        identity=NavixRolloutIdentity(
            env_id=env_id,
            seed=seed,
            node_id=node_id,
            checkpoint_step=_CHECKPOINT_STEP,
            global_step=128,
            algorithm_name="ppo.full_jax.navix",
        ),
        action_names=_ACTION_NAMES,
    )
    return CheckpointInferenceResult(
        trace=trace,
        summaries=(summary,),
        action_names=_ACTION_NAMES,
        source_node_id=node_id,
        checkpoint_step=_CHECKPOINT_STEP,
        checkpoint_kind="ppo.full_jax.navix",
        checkpoint_version=1,
        algorithm_name="ppo.full_jax.navix",
        global_step=128,
        optimizer_updates=16,
        seeds=(seed,),
        max_steps=3,
        rollout_seconds=0.25,
        transfer_seconds=0.01,
    )


def _trace(*, record_video: bool) -> NavixTraceArrays:
    states = np.zeros((1, 4, 4, 4, 3), dtype=np.int32)
    positions = np.asarray(
        [[[1, 1], [1, 1], [1, 2], [1, 2]]],
        dtype=np.int32,
    )
    return NavixTraceArrays(
        full_symbolic=states,
        first_person_symbolic=np.zeros((1, 4, 3, 3, 3), dtype=np.int32),
        policy_observations=np.zeros((1, 4, 8), dtype=np.float32),
        player_positions=positions,
        player_directions=np.asarray([[0, 3, 3, 3]], dtype=np.int32),
        player_pockets=-np.ones((1, 4), dtype=np.int32),
        rgb_frames=(np.zeros((1, 4, 4, 4, 3), dtype=np.uint8) if record_video else None),
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
