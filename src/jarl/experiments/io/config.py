"""Resolved config persistence for node workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from jarl.config import BaseConfig
from jarl.utils import write_text_atomic

__all__ = [
    "save_resolved_config",
]


def save_resolved_config(path: Path, config: BaseConfig) -> Path:
    """Persist a fully resolved config snapshot for a node.

    The caller is responsible for resolving overrides before invoking this
    helper. ``NodeWorkspace`` does not depend on ``ExperimentGraph`` for
    config resolution.

    Args:
        path: Destination path for the config snapshot (typically ``config.json``).
        config: Resolved configuration to persist.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.model_dump(), indent=2, default=str)
    return write_text_atomic(path, payload)
