"""Extension contract and helpers for ``NodeWorkspace`` subclasses.

Subclasses may override training lifecycle hooks in ``NodeWorkspace`` and
delegate persistence to the core workspace APIs:

- Metrics: override is rarely needed; use ``log_scalar`` or JIT callbacks.
- Checkpoints: call ``save_checkpoint`` / ``save_checkpoint_if_due`` from the
  host loop; do not perform Orbax IO inside ``jax.jit``.
- Model archives: call ``save_model_archive`` or the JIT callback helper from
  the host side after extracting pytrees.
- Artifacts: use ``register_artifact`` or ``_register_artifact`` from subclass
  hooks to append entries to ``artifacts.json`` without duplicating registry IO.

Reconstruction from disk uses ``NodeMetadata.workspace_cls``. Invalid import
paths fail by default via ``resolve_workspace_class`` so agent-specific logic
is not silently dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarl.experiments.node import NodeWorkspace

__all__ = [
    "resolve_workspace_class",
]

_DEFAULT_BASE: type[NodeWorkspace] | None = None


def _base_cls() -> type[NodeWorkspace]:
    from jarl.experiments.node import NodeWorkspace

    return NodeWorkspace


def resolve_workspace_class(
    import_path: str,
    *,
    allow_fallback: bool = False,
    base_cls: type[NodeWorkspace] | None = None,
) -> type[NodeWorkspace]:
    """Resolve a workspace class from metadata stored in ``node.json``.

    Invalid import paths raise by default so agent-specific logic is not
    silently dropped. Pass ``allow_fallback=True`` only when explicitly
    tolerating a downgrade to the base ``NodeWorkspace`` class.

    Args:
        import_path: Fully qualified class path from node metadata.
        allow_fallback: Whether to return ``base_cls`` when resolution fails.
        base_cls: Base workspace type used for validation and fallback.

    Returns:
        Resolved workspace class.

    Raises:
        ImportError: If the import path is invalid and fallback is disabled.
        TypeError: If the resolved object is not a workspace subclass and
            fallback is disabled.
    """
    cls = base_cls or _base_cls()
    if not import_path:
        return cls
    try:
        return cls.from_import_path(import_path)
    except (ImportError, TypeError):
        if allow_fallback:
            return cls
        raise
