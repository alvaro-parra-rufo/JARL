"""Read the configurable reward mix of an experiment node."""

from __future__ import annotations

from dataclasses import dataclass

from jarl.envs.navix.reward_config import RewardWeightsConfig
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig

__all__ = ["RewardRequest", "RewardResponse", "reward"]


@dataclass(frozen=True, slots=True)
class RewardRequest:
    """Node whose configurable reward mix to read."""

    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class RewardResponse:
    """Configurable reward mix of a node.

    ``reward`` is ``None`` when the environment uses its native reward.
    """

    node_id: str
    reward: dict[str, float] | None

    def to_compact_dict(self) -> dict[str, object]:
        """Return the mix catalog, including ``reward: null`` for native reward."""
        return {"node_id": self.node_id, "reward": self.reward}


def reward(
    graph: ExperimentGraph[RLRunConfig],
    request: RewardRequest,
) -> RewardResponse:
    """Return the configurable reward mix resolved on a node."""
    workspace = graph.get_node(request.node_id) if request.node_id else graph.current_node
    mix = graph.resolve_config(workspace).environment.reward
    return RewardResponse(node_id=workspace.id, reward=_dump_mix(mix))


def _dump_mix(weights: RewardWeightsConfig | None) -> dict[str, float] | None:
    if weights is None:
        return None
    return {name: float(value) for name, value in weights.model_dump().items()}
