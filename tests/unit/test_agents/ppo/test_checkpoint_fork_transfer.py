"""Integration tests for typed checkpoint fork and transfer."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

from jarl.agents.ppo.checkpoint_state import (
    CHECKPOINT_KIND_PPO,
    CHECKPOINT_KIND_PPO_GRU,
    PPOCheckpoint,
    PPOGrUCheckpoint,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.io.checkpoints import CheckpointRef
from jarl.experiments.io.model_archive import load_model_archive
from jarl.training.config import AlgorithmConfig, EnvironmentConfig, RLRunConfig


def _train_state(params: dict[str, jnp.ndarray]) -> TrainState:
    return TrainState.create(apply_fn=lambda *_args, **_kwargs: None, params=params, tx=optax.adam(1e-3))


class TestTypedCheckpointFork:
    """Fork workflows using typed training checkpoints."""

    def test_fork_restores_parent_ppo_checkpoint(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=RLRunConfig(), branch="main", label="root")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        checkpoint = PPOCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO,
            global_step=5,
            optimizer_updates=2,
            rng_key=jax.random.PRNGKey(9),
            policy_state=_train_state({"policy": jnp.array([1.0])}),
            critic_state=_train_state({"critic": jnp.array([2.0])}),
        )
        root.save_training_checkpoint(5, checkpoint, force=True)
        graph.fork(
            "branch_b",
            from_node=root,
            from_checkpoint=CheckpointRef(node_id=root.id, checkpoint_step=5),
        )
        graph.save()

        child = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig).head("branch_b")
        restored = PPOCheckpoint.from_pytree(child.load_checkpoint())  # type: ignore[arg-type]

        assert restored.policy.params["policy"] == pytest.approx(jnp.array([1.0]))
        assert restored.critic.params["critic"] == pytest.approx(jnp.array([2.0]))

    def test_model_archive_manifest_is_params_only(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "exp"
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=RLRunConfig(), branch="main", label="root")
        root.bind_model_archive_context(
            algorithm_name="ppo.full_jax.navix",
            env_id="Navix-Empty-5x5-v0",
        )
        policy_state = _train_state({"policy": jnp.array([3.0])})
        critic_state = _train_state({"critic": jnp.array([4.0])})
        archive_path = root.save_model_archive(
            "latest",
            3,
            policy=policy_state.params,
            critic=critic_state.params,
            algorithm_name="ppo.full_jax.navix",
            env_id="Navix-Empty-5x5-v0",
        )

        restored = load_model_archive(archive_path)
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))

        assert manifest["algorithm_name"] == "ppo.full_jax.navix"
        assert manifest["env_id"] == "Navix-Empty-5x5-v0"
        assert restored["policy"]["policy"] == pytest.approx(jnp.array([3.0]))
        assert set(restored.keys()) == {"policy", "critic"}

    @pytest.mark.parametrize("action", ["fork", "extend"])
    def test_gru_child_discards_nonzero_carry(self, tmp_path: Path, action: str) -> None:
        exp_dir = tmp_path / "exp"
        base = RLRunConfig(
            environment=EnvironmentConfig(env_id="Navix-Empty-5x5-v0", seed=1, nr_envs=2),
            algorithm=AlgorithmConfig(
                name=CHECKPOINT_KIND_PPO_GRU,
                nr_steps=8,
                minibatch_size=16,
            ),
        )
        graph: ExperimentGraph[RLRunConfig] = ExperimentGraph(exp_dir)
        root = graph.create_root(config=base, branch="main", label="root")
        root.init_checkpoint_manager(max_to_keep=None, save_interval_steps=1)
        carry = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        policy_state = _train_state({"policy": jnp.array([1.0])})
        critic_state = _train_state({"critic": jnp.array([2.0])})
        checkpoint = PPOGrUCheckpoint.build(
            algorithm_name=CHECKPOINT_KIND_PPO_GRU,
            global_step=2,
            optimizer_updates=1,
            rng_key=jax.random.PRNGKey(1),
            policy_state=policy_state,
            critic_state=critic_state,
            policy_carry=carry,
            critic_carry=carry,
        )
        root.save_training_checkpoint(2, checkpoint, force=True)
        ckpt_ref = CheckpointRef(node_id=root.id, checkpoint_step=2)
        if action == "fork":
            graph.fork("continue", from_node=root, from_checkpoint=ckpt_ref)
            child_branch = "continue"
        else:
            graph.extend(from_checkpoint=ckpt_ref)
            child_branch = "main"
        graph.save()

        child = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig).head(child_branch)
        restored = PPOGrUCheckpoint.from_pytree(child.load_parent_checkpoint())  # type: ignore[arg-type]
        zero_carry = jnp.zeros_like(carry)
        _, restored_policy, restored_critic, restored_policy_carry, restored_critic_carry = (
            restored.restore_train_states(
                policy_template=_train_state({"policy": jnp.array([0.0])}),
                critic_template=_train_state({"critic": jnp.array([0.0])}),
                zero_carry=zero_carry,
            )
        )

        assert jnp.allclose(restored_policy_carry, zero_carry)
        assert jnp.allclose(restored_critic_carry, zero_carry)
        assert restored_policy.params["policy"] == pytest.approx(jnp.array([1.0]))
        assert restored_critic.params["critic"] == pytest.approx(jnp.array([2.0]))
