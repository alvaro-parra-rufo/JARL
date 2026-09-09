"""Tests for checkpoint video delegation to common inference."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jarl.agents.ppo.inference.render as render_module
from jarl import env_setup
from jarl.envs.navix.telemetry import NavixTraceArrays
from jarl.experiments.node import NodeMetadata, NodeWorkspace
from jarl.inference.run import CheckpointInferenceRequest, CheckpointInferenceResult
from tests.helpers.minimal_navix_configs import minimal_ppo_run_config


def test_parent_checkpoint_source_uses_fork_origin(
    tmp_path: Path,
) -> None:
    parent = NodeWorkspace(
        node_dir=tmp_path / "parent",
        node_metadata=NodeMetadata(id="main_parent_ab12cd34"),
    )
    child = NodeWorkspace(
        node_dir=tmp_path / "child",
        node_metadata=NodeMetadata(
            id="branch_child_cd34ef56",
            parent_id=parent.id,
            parent_checkpoint_step=12,
        ),
    )
    graph = SimpleNamespace(get_node=lambda node_id: parent if node_id == parent.id else child)

    source, step = render_module._resolve_checkpoint_source(
        graph,
        child,
        checkpoint_step=99,
        load_from_node_id=None,
        use_parent_checkpoint=True,
    )

    assert source is parent
    assert step == 12


def test_render_delegates_multi_episode_capture_to_common_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = NodeWorkspace(
        node_dir=tmp_path / "target",
        node_metadata=NodeMetadata(id="main_target_ab12cd34"),
    )
    source = NodeWorkspace(
        node_dir=tmp_path / "source",
        node_metadata=NodeMetadata(id="main_source_cd34ef56"),
    )
    config = minimal_ppo_run_config().apply_overrides(
        {
            "environment.seed": 5,
            "video.final_video_episodes": 2,
            "video.video_view_mode": "first_person",
        }
    )

    class FakeGraph:
        def get_node(self, node_id: str) -> NodeWorkspace:
            return source if node_id == source.id else target

        def resolve_config(self, workspace: NodeWorkspace) -> object:
            assert workspace is target
            return config

    fake_graph = FakeGraph()
    monkeypatch.setattr(
        render_module,
        "ExperimentGraph",
        SimpleNamespace(from_directory=lambda *_args, **_kwargs: fake_graph),
    )
    monkeypatch.setattr(env_setup, "setup_jax", lambda _config: None)
    monkeypatch.setattr(
        env_setup,
        "log_jax_training_devices",
        lambda _logger: None,
    )
    requests: list[CheckpointInferenceRequest] = []
    fake_result = _video_result(source.id)

    def run_inference(
        source_workspace: NodeWorkspace,
        target_config: object,
        request: CheckpointInferenceRequest,
    ) -> CheckpointInferenceResult:
        assert source_workspace is source
        assert target_config.environment.env_id == "Navix-Empty-5x5-v0"  # type: ignore[attr-defined]
        requests.append(request)
        return fake_result

    monkeypatch.setattr(
        render_module,
        "run_checkpoint_inference",
        run_inference,
    )
    encoded: list[CheckpointInferenceResult] = []

    def encode(**kwargs: object) -> list[Path]:
        encoded.append(kwargs["result"])  # type: ignore[arg-type]
        return [tmp_path / "video.mp4"]

    monkeypatch.setattr(render_module, "_encode_checkpoint_video", encode)

    paths = render_module.render_checkpoint_greedy_video(
        experiment_dir=tmp_path,
        node_id=target.id,
        checkpoint_step=9,
        env_id="Navix-Empty-5x5-v0",
        role="final",
        name_prefix="final",
        load_from_node_id=source.id,
    )

    assert paths == [tmp_path / "video.mp4"]
    assert len(requests) == 1
    request = requests[0]
    assert request.checkpoint_step == 9
    assert request.seeds == (5, 6)
    assert request.summarize is False
    assert request.capture_profile.rgb_view_mode == "first_person"
    assert encoded == [fake_result]


def _video_result(source_node_id: str) -> CheckpointInferenceResult:
    """Build a minimal two-episode host video result."""
    trace = NavixTraceArrays(
        full_symbolic=None,
        first_person_symbolic=None,
        policy_observations=None,
        player_positions=None,
        player_directions=None,
        player_pockets=None,
        rgb_frames=np.zeros((2, 3, 4, 4, 3), dtype=np.uint8),
        actions=np.zeros((2, 2), dtype=np.int32),
        rewards=np.zeros((2, 2), dtype=np.float32),
        step_types=np.zeros((2, 2), dtype=np.int32),
        dones=np.zeros((2, 2), dtype=np.bool_),
        active_mask=np.ones((2, 2), dtype=np.bool_),
        event_happened=None,
        event_positions=None,
        event_colours=None,
        event_types=None,
        mission_present=np.zeros((2,), dtype=np.bool_),
        mission_position=-np.ones((2, 2), dtype=np.int32),
        mission_colour=-np.ones((2,), dtype=np.int32),
        mission_event_type=-np.ones((2,), dtype=np.int32),
        episode_returns=np.zeros((2,), dtype=np.float32),
        episode_lengths=np.full((2,), 2, dtype=np.int32),
        episode_done=np.zeros((2,), dtype=np.bool_),
    )
    return CheckpointInferenceResult(
        trace=trace,
        summaries=(),
        action_names=("rotate_ccw",),
        source_node_id=source_node_id,
        checkpoint_step=9,
        checkpoint_kind="ppo.full_jax.navix",
        checkpoint_version=1,
        algorithm_name="ppo.full_jax.navix",
        global_step=128,
        optimizer_updates=16,
        seeds=(5, 6),
        max_steps=2,
        rollout_seconds=0.1,
        transfer_seconds=0.01,
    )
