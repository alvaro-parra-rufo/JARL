"""Internal helpers for train operations."""

from __future__ import annotations

from typing import Any

from jarl.training.config import RLRunConfig
from jarl.training.presets import RunFormPayload, run_overrides_from_payload
from jarl.training.run_overrides import ensure_validated_config_overrides

__all__ = ["resolve_train_overrides"]


def resolve_train_overrides(
    base_config: RLRunConfig,
    *,
    payload: RunFormPayload | None,
    config_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve and validate sparse overrides against ``base_config``."""
    if payload is not None:
        return run_overrides_from_payload(payload, base_config=base_config)
    if not config_overrides:
        return None
    ensure_validated_config_overrides(base_config, config_overrides, policy="retrain")
    return config_overrides
