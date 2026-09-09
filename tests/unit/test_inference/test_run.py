"""Tests for common checkpoint inference orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jarl.inference.run as inference_run
from jarl.envs.navix.telemetry import NavixCaptureProfile, NavixTraceArrays
from jarl.experiments.node import NodeMetadata, NodeWorkspace
from jarl.inference.policy import (
    InferencePolicyRuntime,
    LoadedInferencePolicy,
    PolicyState,
)
from jarl.inference.run import (
    CheckpointInferenceRequest,
    run_checkpoint_inference,
)
from tests.helpers.minimal_navix_configs import minimal_ppo_run_config


class TestCheckpointInferenceRequest:
    """Request validation rejects non-executable batches."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param(
                {"checkpoint_step": -1, "seeds": (0,)},
                "checkpoint_step",
                id="checkpoint",
            ),
            pytest.param(
                {"checkpoint_step": 0, "seeds": ()},
                "At least one",
                id="seeds",
            ),
            pytest.param(
                {"checkpoint_step": 0, "seeds": (0,), "max_steps": 0},
                "max_steps",
                id="horizon",
            ),
        ],
    )
    def test_invalid_request_raises(
        self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CheckpointInferenceRequest(
                capture_profile=NavixCaptureProfile.analysis(),
                **kwargs,  # type: ignore[arg-type]
            )

    def test_video_profile_requires_summary_opt_out(self) -> None:
        with pytest.raises(ValueError, match="Summarization requires"):
            CheckpointInferenceRequest(
                checkpoint_step=0,
                seeds=(0,),
                capture_profile=NavixCaptureProfile.video(),
            )


class TestRunCheckpointInference:
    """The service restores once, batches seeds, and transfers one trace."""

    def test_runs_multiple_direct_seeds_in_one_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        source = NodeWorkspace(
            node_dir=tmp_path / "source",
            node_metadata=NodeMetadata(id="main_source_ab12cd34"),
        )
        config = minimal_ppo_run_config().apply_overrides({"environment.max_episode_steps": 2})
        backend_calls: list[tuple[str, int]] = []

        def backend(
            workspace: NodeWorkspace,
            target_config: object,
            checkpoint_step: int,
            env: object,
        ) -> LoadedInferencePolicy:
            del target_config
            backend_calls.append((workspace.id, checkpoint_step))

            def initial_state(batch_size: int) -> jax.Array:
                return jnp.zeros((batch_size,), dtype=jnp.int32)

            def greedy_step(
                params: object,
                observation: jax.Array,
                state: PolicyState,
            ) -> tuple[jax.Array, PolicyState]:
                del params
                return (
                    jnp.zeros((observation.shape[0],), dtype=jnp.int32),
                    jnp.asarray(state),
                )

            runtime = InferencePolicyRuntime(
                preprocess_observation=env.preprocess_observation,  # type: ignore[attr-defined]
                initial_state=initial_state,
                greedy_step=greedy_step,
            )
            return LoadedInferencePolicy(
                runtime=runtime,
                params=(),
                source_node_id=workspace.id,
                checkpoint_step=checkpoint_step,
                checkpoint_kind="ppo.full_jax.navix",
                checkpoint_version=1,
                algorithm_name="ppo.full_jax.navix",
                global_step=128,
                optimizer_updates=16,
            )

        monkeypatch.setattr(
            inference_run,
            "resolve_inference_backend_from_name",
            lambda _name: backend,
        )
        result = run_checkpoint_inference(
            source,
            config,
            CheckpointInferenceRequest(
                checkpoint_step=4,
                seeds=(7, 11),
                capture_profile=NavixCaptureProfile.analysis(),
                max_steps=2,
            ),
        )

        assert backend_calls == [(source.id, 4)]
        assert result.seeds == (7, 11)
        assert result.trace.actions.shape == (2, 2)
        np.testing.assert_array_equal(result.trace.actions, np.zeros((2, 2)))
        assert len(result.summaries) == 2
        assert tuple(summary.identity.seed for summary in result.summaries) == (
            7,
            11,
        )
        assert result.global_step == 128

    def test_video_capture_uses_one_rollout_and_host_transfer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        source = NodeWorkspace(
            node_dir=tmp_path / "source",
            node_metadata=NodeMetadata(id="main_source_ab12cd34"),
        )
        config = minimal_ppo_run_config()
        host_trace = _video_trace()
        transfers = 0
        rollout_calls = 0

        class DeviceTrace:
            actions = jnp.zeros((1, 2), dtype=jnp.int32)

            def to_host(self) -> NavixTraceArrays:
                nonlocal transfers
                transfers += 1
                return host_trace

        class FakeRollout:
            action_names = ("rotate_ccw",)

            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def rollout(self, params: object, keys: jax.Array) -> DeviceTrace:
                nonlocal rollout_calls
                del params
                rollout_calls += 1
                assert keys.shape == (1, 2)
                return DeviceTrace()

        runtime = InferencePolicyRuntime(
            preprocess_observation=lambda observation: observation,
            initial_state=lambda _batch_size: (),
            greedy_step=lambda _params, observation, state: (
                jnp.zeros((observation.shape[0],), dtype=jnp.int32),
                state,
            ),
        )
        loaded = LoadedInferencePolicy(
            runtime=runtime,
            params=(),
            source_node_id=source.id,
            checkpoint_step=4,
            checkpoint_kind="ppo.full_jax.navix",
            checkpoint_version=1,
            algorithm_name="ppo.full_jax.navix",
            global_step=128,
            optimizer_updates=16,
        )
        monkeypatch.setattr(
            inference_run,
            "navix_full_jit_env_factory",
            lambda _config: SimpleNamespace(horizon=2),
        )
        monkeypatch.setattr(
            inference_run,
            "resolve_inference_backend_from_name",
            lambda _name: lambda _workspace, _config, _step, _env: loaded,
        )
        monkeypatch.setattr(inference_run, "NavixTelemetryRollout", FakeRollout)

        result = run_checkpoint_inference(
            source,
            config,
            CheckpointInferenceRequest(
                checkpoint_step=4,
                seeds=(7,),
                capture_profile=NavixCaptureProfile.video(),
                max_steps=2,
                summarize=False,
            ),
        )

        assert result.trace is host_trace
        assert rollout_calls == 1
        assert transfers == 1


def _video_trace() -> NavixTraceArrays:
    """Return a minimal host trace for one video episode."""
    return NavixTraceArrays(
        full_symbolic=None,
        first_person_symbolic=None,
        policy_observations=None,
        player_positions=None,
        player_directions=None,
        player_pockets=None,
        rgb_frames=np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
        actions=np.zeros((1, 2), dtype=np.int32),
        rewards=np.zeros((1, 2), dtype=np.float32),
        step_types=np.zeros((1, 2), dtype=np.int32),
        dones=np.zeros((1, 2), dtype=np.bool_),
        active_mask=np.ones((1, 2), dtype=np.bool_),
        event_happened=None,
        event_positions=None,
        event_colours=None,
        event_types=None,
        mission_present=np.zeros((1,), dtype=np.bool_),
        mission_position=-np.ones((1, 2), dtype=np.int32),
        mission_colour=-np.ones((1,), dtype=np.int32),
        mission_event_type=-np.ones((1,), dtype=np.int32),
        episode_returns=np.zeros((1,), dtype=np.float32),
        episode_lengths=np.full((1,), 2, dtype=np.int32),
        episode_done=np.zeros((1,), dtype=np.bool_),
    )
