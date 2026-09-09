"""Patch the configurable reward mix of an unused experiment node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.training.config import RLRunConfig
from jarl.utils import flatten_dict, unflatten_dict

__all__ = ["SetRewardRequest", "SetRewardResponse", "set_reward"]

_VIRGIN_STATUSES = frozenset({NodeStatus.CREATED, NodeStatus.PREPARED})
_CHANNEL_NAMES = frozenset(RewardWeightsConfig.model_fields)
_TRAINED_NODE_ERROR = (
    "Cannot change the reward mix on a node that has already trained. "
    "Fork or extend a new node, then set the mix there."
)
_CHECKPOINT_NODE_ERROR = (
    "Cannot change the reward mix on a node that already has checkpoints. "
    "Fork or extend a new node, then set the mix there."
)
_EMPTY_PATCH_ERROR = "Provide at least one reward channel to update."
_INVALID_WEIGHT_ERROR = "Reward weights must be finite and greater than or equal to 0."


@dataclass(frozen=True, slots=True)
class SetRewardRequest:
    """Sparse patch of configurable reward weights."""

    weights: dict[str, float]
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class SetRewardResponse:
    """Configurable reward mix after an in-place patch."""

    node_id: str
    status: str
    reward: dict[str, float]

    def to_compact_dict(self) -> dict[str, object]:
        """Return the node identity and the resolved configurable mix."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "reward": dict(self.reward),
        }


def set_reward(
    graph: ExperimentGraph[RLRunConfig],
    request: SetRewardRequest,
) -> SetRewardResponse:
    """Merge a sparse reward patch into an unused node and persist it in place."""
    workspace = graph.get_node(request.node_id) if request.node_id else graph.current_node
    _require_unused_node(workspace)
    patch = _validated_patch(request.weights)
    current = graph.resolve_config(workspace)
    merged = _merge_weights(current.environment.reward, patch)
    new_config = current.model_copy(
        update={"environment": current.environment.model_copy(update={"reward": merged})},
    )
    workspace.save_config_overrides(_with_reward_overrides(workspace.node_metadata.config_overrides, merged))
    workspace.save_resolved_config(new_config)
    graph.save()
    return SetRewardResponse(
        node_id=workspace.id,
        status=workspace.status.value,
        reward={name: float(value) for name, value in merged.model_dump().items()},
    )


def _require_unused_node(workspace: NodeWorkspace) -> None:
    if workspace.status not in _VIRGIN_STATUSES:
        raise ValueError(_TRAINED_NODE_ERROR)
    if workspace.list_checkpoints():
        raise ValueError(_CHECKPOINT_NODE_ERROR)


def _validated_patch(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError(_EMPTY_PATCH_ERROR)
    unknown = set(weights) - _CHANNEL_NAMES
    if unknown:
        raise ValueError(f"Unknown reward channels: {', '.join(sorted(unknown))}.")
    return dict(weights)


def _merge_weights(current: RewardWeightsConfig | None, patch: dict[str, float]) -> RewardWeightsConfig:
    base = RewardWeightsConfig() if current is None else current
    try:
        return RewardWeightsConfig.model_validate({**base.model_dump(), **patch})
    except ValidationError as exc:
        raise ValueError(_INVALID_WEIGHT_ERROR) from exc


def _with_reward_overrides(
    existing: dict[str, Any],
    weights: RewardWeightsConfig,
) -> dict[str, Any]:
    nested = unflatten_dict(dict(existing))
    environment = dict(nested.get("environment") or {})
    environment["reward"] = weights.model_dump()
    nested["environment"] = environment
    return flatten_dict(nested)
