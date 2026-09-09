"""Tests for algorithm-independent inference policy contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jarl.inference.policy import InferencePolicyRuntime


class TestInferencePolicyRuntime:
    """Runtime functions remain reusable across parameter pytrees."""

    def test_greedy_step_accepts_different_params(self) -> None:
        runtime = InferencePolicyRuntime(
            preprocess_observation=lambda observation: observation,
            initial_state=lambda batch_size: jnp.zeros((batch_size,), dtype=jnp.int32),
            greedy_step=lambda params, observation, state: (
                observation + params,
                state + 1,
            ),
        )
        observation = jnp.asarray([2], dtype=jnp.int32)
        state = runtime.initial_state(1)
        compiled_step = jax.jit(runtime.greedy_step)

        first_action, first_state = compiled_step(jnp.asarray([1]), observation, state)
        second_action, second_state = compiled_step(jnp.asarray([4]), observation, state)

        assert first_action.tolist() == [3]
        assert second_action.tolist() == [6]
        assert first_state.tolist() == [1]
        assert second_state.tolist() == [1]
