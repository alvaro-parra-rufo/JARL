"""Compare resolved configs between two experiment nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.config import ConfigDiff
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.summaries import config_highlights
from jarl.training.config import RLRunConfig

__all__ = ["DiffRequest", "DiffResponse", "diff"]

_HIDDEN_ENVIRONMENT_KEYS = frozenset(
    {
        "reward",
        "scenario_reward_id",
        "scenario_reward_version",
    }
)
"""Environment fields omitted from the agent-facing compact diff."""


def _is_hidden_diff_key(key: str) -> bool:
    return (
        key == "environment.reward"
        or key.startswith("environment.reward.")
        or key.startswith("environment.scenario_reward_")
    )


def _strip_reward_payload(value: object) -> object:
    if isinstance(value, dict):
        return {
            nested_key: _strip_reward_payload(nested_value)
            for nested_key, nested_value in value.items()
            if nested_key not in _HIDDEN_ENVIRONMENT_KEYS
        }
    if isinstance(value, tuple):
        try:
            baseline, target = value
        except ValueError:
            return value
        return (_strip_reward_payload(baseline), _strip_reward_payload(target))
    return value


def _filter_diff_section(section: dict[str, object]) -> dict[str, object]:
    filtered: dict[str, object] = {}
    for key, value in section.items():
        if _is_hidden_diff_key(key):
            continue
        stripped = _strip_reward_payload(value)
        if isinstance(stripped, tuple):
            try:
                baseline, target = stripped
            except ValueError:
                filtered[key] = stripped
                continue
            if baseline == target:
                continue
        filtered[key] = stripped
    return filtered


@dataclass(frozen=True, slots=True)
class DiffRequest:
    """Nodes to compare."""

    node_a: str
    node_b: str


@dataclass(frozen=True, slots=True)
class DiffResponse:
    """Outcome of ``diff``."""

    node_a: str
    node_b: str
    diff: ConfigDiff
    config_a_highlights: dict[str, str | int | float | bool]
    config_b_highlights: dict[str, str | int | float | bool]

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        changed = _filter_diff_section(dict(self.diff.changed))
        return {
            "node_a": self.node_a,
            "node_b": self.node_b,
            "diff": {
                "added": _filter_diff_section(self.diff.added),
                "removed": _filter_diff_section(self.diff.removed),
                "changed": {key: list(values) for key, values in changed.items() if isinstance(values, tuple)},
            },
            "config_a_highlights": self.config_a_highlights,
            "config_b_highlights": self.config_b_highlights,
        }


def diff(
    graph: ExperimentGraph[RLRunConfig],
    request: DiffRequest,
) -> DiffResponse:
    """Return the config diff and highlights for two nodes."""
    workspace_a = graph.get_node(request.node_a)
    workspace_b = graph.get_node(request.node_b)
    config_a = graph.resolve_config(workspace_a)
    config_b = graph.resolve_config(workspace_b)
    config_diff = graph.get_config_diff(workspace_a, workspace_b)
    return DiffResponse(
        node_a=request.node_a,
        node_b=request.node_b,
        diff=config_diff,
        config_a_highlights=config_highlights(config_a),
        config_b_highlights=config_highlights(config_b),
    )
