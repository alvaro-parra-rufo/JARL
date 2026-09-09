"""Tests for inference backend discovery."""

from __future__ import annotations

import pytest

from jarl.agents.ppo.inference.backend import (
    load_ppo_gru_inference_policy,
    load_ppo_inference_policy,
)
from jarl.inference.registry import (
    list_inference_backend_names,
    resolve_inference_backend_from_name,
)


class TestInferenceBackendRegistry:
    """Inference backends resolve lazily from algorithm names."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("ppo.full_jax.navix", load_ppo_inference_policy, id="ppo"),
            pytest.param("ppo_gru.full_jax.navix", load_ppo_gru_inference_policy, id="ppo_gru"),
        ],
    )
    def test_resolve_known_backend(self, name: str, expected: object) -> None:
        backend = resolve_inference_backend_from_name(name)

        assert backend is expected

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown algorithm\.name"):
            resolve_inference_backend_from_name("unknown.agent")

    def test_list_backend_names_is_stable(self) -> None:
        names = list_inference_backend_names()

        assert names == ("ppo.full_jax.navix", "ppo_gru.full_jax.navix")
