"""PPO and PPO-GRU training checkpoint payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jarl.experiments.checkpoint_state import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingCheckpoint,
    TrainStateLeaf,
    deserialize_rng_key,
    restore_train_state,
    serialize_rng_key,
    train_state_to_leaf,
)

__all__ = [
    "CHECKPOINT_IDS_TO_KIND",
    "CHECKPOINT_KIND_IDS",
    "CHECKPOINT_KIND_PPO",
    "CHECKPOINT_KIND_PPO_GRU",
    "CHECKPOINT_TYPES",
    "PPOCheckpoint",
    "PPOGrUCheckpoint",
    "checkpoint_from_pytree",
    "checkpoint_kind_for_algorithm",
]

CHECKPOINT_KIND_PPO = "ppo.full_jax.navix"
CHECKPOINT_KIND_PPO_GRU = "ppo_gru.full_jax.navix"

CHECKPOINT_KIND_IDS: dict[str, int] = {
    CHECKPOINT_KIND_PPO: 1,
    CHECKPOINT_KIND_PPO_GRU: 2,
}
CHECKPOINT_IDS_TO_KIND: dict[int, str] = {value: key for key, value in CHECKPOINT_KIND_IDS.items()}

CHECKPOINT_TYPES: dict[str, type[TrainingCheckpoint]] = {}


@dataclass(frozen=True, slots=True)
class PPOCheckpoint:
    """Feedforward PPO checkpoint payload."""

    version: int
    kind: str
    algorithm_name: str
    global_step: int
    optimizer_updates: int
    rng_key: tuple[int, ...]
    policy: TrainStateLeaf
    critic: TrainStateLeaf

    @classmethod
    def build(
        cls,
        *,
        algorithm_name: str,
        global_step: int,
        optimizer_updates: int,
        rng_key: jax.Array,
        policy_state: TrainState,
        critic_state: TrainState,
    ) -> PPOCheckpoint:
        """Construct a checkpoint from live training state."""
        return cls(
            version=CHECKPOINT_FORMAT_VERSION,
            kind=CHECKPOINT_KIND_PPO,
            algorithm_name=algorithm_name,
            global_step=int(global_step),
            optimizer_updates=int(optimizer_updates),
            rng_key=serialize_rng_key(rng_key),
            policy=train_state_to_leaf(policy_state),
            critic=train_state_to_leaf(critic_state),
        )

    def to_pytree(self) -> dict[str, Any]:
        """Serialize the checkpoint into an Orbax-compatible dict pytree."""
        return {
            "kind_id": jnp.asarray(CHECKPOINT_KIND_IDS[self.kind], dtype=jnp.int32),
            "version": jnp.asarray(self.version, dtype=jnp.int32),
            "global_step": jnp.asarray(self.global_step, dtype=jnp.int32),
            "optimizer_updates": jnp.asarray(self.optimizer_updates, dtype=jnp.int32),
            "rng_key": jnp.asarray(self.rng_key, dtype=jnp.uint32),
            "policy_step": jnp.asarray(self.policy.step, dtype=jnp.int32),
            "policy_params": self.policy.params,
            "policy_opt_state": self.policy.opt_state,
            "critic_step": jnp.asarray(self.critic.step, dtype=jnp.int32),
            "critic_params": self.critic.params,
            "critic_opt_state": self.critic.opt_state,
        }

    @classmethod
    def from_pytree(cls, tree: dict[str, Any]) -> PPOCheckpoint:
        """Reconstruct a checkpoint from an Orbax-restored dict pytree."""
        kind = _kind_from_tree(tree)
        if kind != CHECKPOINT_KIND_PPO:
            msg = f"Expected PPO checkpoint kind {CHECKPOINT_KIND_PPO!r}, got {kind!r}."
            raise ValueError(msg)
        return cls(
            kind=CHECKPOINT_KIND_PPO,
            algorithm_name=CHECKPOINT_KIND_PPO,
            **_checkpoint_fields_from_tree(tree),
        )

    def restore_train_states(
        self,
        *,
        policy_template: TrainState,
        critic_template: TrainState,
    ) -> tuple[jax.Array, TrainState, TrainState]:
        """Restore PRNG key and train states from this checkpoint."""
        key = deserialize_rng_key(self.rng_key)
        policy_state = restore_train_state(policy_template, self.policy)
        critic_state = restore_train_state(critic_template, self.critic)
        return key, policy_state, critic_state


@dataclass(frozen=True, slots=True)
class PPOGrUCheckpoint:
    """Recurrent PPO-GRU checkpoint payload with policy and critic GRU carries."""

    version: int
    kind: str
    algorithm_name: str
    global_step: int
    optimizer_updates: int
    rng_key: tuple[int, ...]
    policy: TrainStateLeaf
    critic: TrainStateLeaf
    policy_carry: Any | None = None
    critic_carry: Any | None = None

    @classmethod
    def build(
        cls,
        *,
        algorithm_name: str,
        global_step: int,
        optimizer_updates: int,
        rng_key: jax.Array,
        policy_state: TrainState,
        critic_state: TrainState,
        policy_carry: jax.Array | None = None,
        critic_carry: jax.Array | None = None,
    ) -> PPOGrUCheckpoint:
        """Construct a checkpoint from live recurrent training state."""
        return cls(
            version=CHECKPOINT_FORMAT_VERSION,
            kind=CHECKPOINT_KIND_PPO_GRU,
            algorithm_name=algorithm_name,
            global_step=int(global_step),
            optimizer_updates=int(optimizer_updates),
            rng_key=serialize_rng_key(rng_key),
            policy=train_state_to_leaf(policy_state),
            critic=train_state_to_leaf(critic_state),
            policy_carry=_host_carry(policy_carry),
            critic_carry=_host_carry(critic_carry),
        )

    def to_pytree(self) -> dict[str, Any]:
        """Serialize the checkpoint into an Orbax-compatible dict pytree."""
        payload = PPOCheckpoint(
            version=self.version,
            kind=self.kind,
            algorithm_name=self.algorithm_name,
            global_step=self.global_step,
            optimizer_updates=self.optimizer_updates,
            rng_key=self.rng_key,
            policy=self.policy,
            critic=self.critic,
        ).to_pytree()
        if self.policy_carry is not None:
            payload["policy_carry"] = jnp.asarray(self.policy_carry)
        if self.critic_carry is not None:
            payload["critic_carry"] = jnp.asarray(self.critic_carry)
        return payload

    @classmethod
    def from_pytree(cls, tree: dict[str, Any]) -> PPOGrUCheckpoint:
        """Reconstruct a checkpoint from an Orbax-restored dict pytree."""
        kind = _kind_from_tree(tree)
        if kind != CHECKPOINT_KIND_PPO_GRU:
            msg = f"Expected GRU checkpoint kind {CHECKPOINT_KIND_PPO_GRU!r}, got {kind!r}."
            raise ValueError(msg)
        return cls(
            kind=CHECKPOINT_KIND_PPO_GRU,
            algorithm_name=CHECKPOINT_KIND_PPO_GRU,
            policy_carry=tree.get("policy_carry"),
            critic_carry=tree.get("critic_carry"),
            **_checkpoint_fields_from_tree(tree),
        )

    def restore_train_states(
        self,
        *,
        policy_template: TrainState,
        critic_template: TrainState,
        zero_carry: jax.Array,
    ) -> tuple[jax.Array, TrainState, TrainState, jax.Array, jax.Array]:
        """Restore PRNG key and train states. GRU hidden state is always reset.

        ``train()`` always calls ``env.reset()``, so episodic GRU carries from the
        checkpoint are discarded even when present for payload compatibility.
        """
        key, policy_state, critic_state = PPOCheckpoint(
            version=self.version,
            kind=self.kind,
            algorithm_name=self.algorithm_name,
            global_step=self.global_step,
            optimizer_updates=self.optimizer_updates,
            rng_key=self.rng_key,
            policy=self.policy,
            critic=self.critic,
        ).restore_train_states(
            policy_template=policy_template,
            critic_template=critic_template,
        )
        return key, policy_state, critic_state, zero_carry, zero_carry


CHECKPOINT_TYPES[CHECKPOINT_KIND_PPO] = PPOCheckpoint
CHECKPOINT_TYPES[CHECKPOINT_KIND_PPO_GRU] = PPOGrUCheckpoint


def _host_carry(carry: jax.Array | None) -> Any | None:
    """Copy a GRU carry to host memory when present."""
    return None if carry is None else jax.device_get(carry)


def _checkpoint_fields_from_tree(tree: dict[str, Any]) -> dict[str, Any]:
    rng_key = tuple(int(value) for value in jnp.asarray(tree["rng_key"]).reshape(-1))
    return {
        "version": int(tree["version"]),
        "global_step": int(tree["global_step"]),
        "optimizer_updates": int(tree["optimizer_updates"]),
        "rng_key": rng_key,
        "policy": TrainStateLeaf(
            step=int(tree["policy_step"]),
            params=tree["policy_params"],
            opt_state=tree["policy_opt_state"],
        ),
        "critic": TrainStateLeaf(
            step=int(tree["critic_step"]),
            params=tree["critic_params"],
            opt_state=tree["critic_opt_state"],
        ),
    }


def checkpoint_kind_for_algorithm(algorithm_name: str) -> str:
    """Return the checkpoint kind associated with an algorithm name."""
    if algorithm_name == CHECKPOINT_KIND_PPO_GRU:
        return CHECKPOINT_KIND_PPO_GRU
    if algorithm_name == CHECKPOINT_KIND_PPO:
        return CHECKPOINT_KIND_PPO
    msg = f"Unsupported algorithm for checkpoint dispatch: {algorithm_name!r}"
    raise ValueError(msg)


def checkpoint_from_pytree(tree: dict[str, Any]) -> TrainingCheckpoint:
    """Dispatch a restored pytree to the concrete checkpoint dataclass."""
    if not isinstance(tree, dict):
        msg = "Checkpoint payload must be a mapping."
        raise TypeError(msg)
    kind = _kind_from_tree(tree)
    checkpoint_cls = CHECKPOINT_TYPES.get(kind)
    if checkpoint_cls is None:
        msg = f"Unknown checkpoint kind: {kind!r}"
        raise ValueError(msg)
    return checkpoint_cls.from_pytree(tree)


def _kind_from_tree(tree: dict[str, Any]) -> str:
    if "kind_id" in tree:
        kind_id = int(jnp.asarray(tree["kind_id"]))
        kind = CHECKPOINT_IDS_TO_KIND.get(kind_id)
        if kind is None:
            msg = f"Unknown checkpoint kind_id: {kind_id}"
            raise ValueError(msg)
        return kind
    legacy_kind = tree.get("kind")
    if isinstance(legacy_kind, str):
        return legacy_kind
    msg = "Checkpoint payload is missing a supported kind identifier."
    raise ValueError(msg)
