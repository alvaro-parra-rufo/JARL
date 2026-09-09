"""Trainer registry keyed by ``AlgorithmConfig.name``."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from jarl.training.trainer import Trainer

__all__ = [
    "TRAINER_REGISTRY",
    "list_trainer_names",
    "resolve_trainer_from_name",
]

TRAINER_REGISTRY: dict[str, tuple[str, str]] = {
    "ppo.full_jax.navix": ("jarl.agents.ppo.trainer", "ppo_full_jax_trainer"),
    "ppo_gru.full_jax.navix": ("jarl.agents.ppo.gru_trainer", "ppo_gru_full_jax_trainer"),
}
"""Mapping from canonical algorithm name to ``(module, callable)`` import path."""


def list_trainer_names() -> tuple[str, ...]:
    """Return registered trainer algorithm names in stable order."""
    return tuple(sorted(TRAINER_REGISTRY))


def resolve_trainer_from_name(name: str) -> Trainer:
    """Resolve a trainer callable from a canonical algorithm name.

    Args:
        name: Value of ``AlgorithmConfig.name`` for the experiment.

    Returns:
        Trainer callable registered for ``name``.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    import_path = TRAINER_REGISTRY.get(name)
    if import_path is None:
        known = ", ".join(list_trainer_names())
        msg = f"Unknown algorithm.name {name!r}. Registered trainers: {known}."
        raise ValueError(msg)
    module_path, qualname = import_path
    module = importlib.import_module(module_path)
    trainer = getattr(module, qualname)
    if not callable(trainer):
        msg = f"Trainer import path for {name!r} is not callable."
        raise TypeError(msg)
    return trainer


def import_trainer_callable(import_path: str) -> Callable[..., object]:
    """Import a trainer from a ``module:callable`` string.

    Args:
        import_path: Import path in ``module:callable`` format.

    Returns:
        Imported callable.

    Raises:
        ValueError: If the import path format is invalid.
        TypeError: If the resolved object is not callable.
    """
    module_path, separator, qualname = import_path.partition(":")
    if separator != ":":
        msg = f"Trainer import path must use module:callable format, got {import_path!r}."
        raise ValueError(msg)
    module = importlib.import_module(module_path)
    obj = getattr(module, qualname)
    if not callable(obj):
        msg = f"Trainer import path {import_path!r} is not callable."
        raise TypeError(msg)
    return obj
