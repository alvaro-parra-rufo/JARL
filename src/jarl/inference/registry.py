"""Inference backend registry keyed by ``AlgorithmConfig.name``."""

from __future__ import annotations

import importlib
from typing import cast

from jarl.inference.backend import InferenceBackend

__all__ = [
    "INFERENCE_BACKEND_REGISTRY",
    "list_inference_backend_names",
    "resolve_inference_backend_from_name",
]

INFERENCE_BACKEND_REGISTRY: dict[str, tuple[str, str]] = {
    "ppo.full_jax.navix": (
        "jarl.agents.ppo.inference.backend",
        "load_ppo_inference_policy",
    ),
    "ppo_gru.full_jax.navix": (
        "jarl.agents.ppo.inference.backend",
        "load_ppo_gru_inference_policy",
    ),
}
"""Mapping from canonical algorithm name to ``(module, callable)`` import path."""


def list_inference_backend_names() -> tuple[str, ...]:
    """Return registered inference algorithm names in stable order."""
    return tuple(sorted(INFERENCE_BACKEND_REGISTRY))


def resolve_inference_backend_from_name(name: str) -> InferenceBackend:
    """Resolve the inference backend registered for an algorithm name.

    Args:
        name: Value of ``AlgorithmConfig.name`` for the experiment.

    Returns:
        Callable that loads an inference policy from a checkpoint.

    Raises:
        TypeError: If the registered object is not callable.
        ValueError: If ``name`` is not registered.
    """
    import_path = INFERENCE_BACKEND_REGISTRY.get(name)
    if import_path is None:
        known = ", ".join(list_inference_backend_names())
        msg = f"Unknown algorithm.name {name!r}. Registered inference backends: {known}."
        raise ValueError(msg)
    module_path, qualname = import_path
    backend = getattr(importlib.import_module(module_path), qualname)
    if not callable(backend):
        msg = f"Inference backend import path for {name!r} is not callable."
        raise TypeError(msg)
    return cast(InferenceBackend, backend)
