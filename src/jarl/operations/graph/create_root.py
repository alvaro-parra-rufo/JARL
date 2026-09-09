"""Create a prepared root node on an experiment graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.contracts.compact import dataclass_to_compact_dict
from jarl.operations.graph._helpers import node_count
from jarl.training.config import RLRunConfig
from jarl.training.presets import (
    AlgorithmChoice,
    PresetKind,
    RunFormPayload,
    default_form_values,
    form_values_to_run_config,
    preset_run_config,
)

__all__ = ["CreateRootRequest", "CreateRootResponse", "create_root"]


@dataclass(frozen=True, slots=True)
class CreateRootRequest:
    """Inputs for creating a prepared root node."""

    label: str
    branch: str = "main"
    config: RLRunConfig | None = None
    preset: PresetKind = "fast"
    algorithm: AlgorithmChoice = "ppo"
    form: RunFormPayload | None = None


@dataclass(frozen=True, slots=True)
class CreateRootResponse:
    """Outcome of ``create_root``."""

    node_id: str
    branch: str
    status: str

    _compact_include: ClassVar[frozenset[str] | None] = None

    def to_compact_dict(self) -> dict[str, object]:
        """Return a compact projection for tools and agents."""
        return dataclass_to_compact_dict(self, include=self._compact_include)


def create_root(
    graph: ExperimentGraph[RLRunConfig],
    request: CreateRootRequest,
) -> CreateRootResponse:
    """Create a root node in ``prepared`` status without training."""
    if node_count(graph) > 0:
        msg = f"Cannot create root when experiment already has {node_count(graph)} node(s)."
        raise ValueError(msg)

    if request.config is not None:
        config = request.config
    elif request.form is not None:
        config = form_values_to_run_config(request.form)
    elif request.preset == "fast":
        config = preset_run_config(algorithm=request.algorithm)
    else:
        config = form_values_to_run_config(
            RunFormPayload(
                values=default_form_values(preset="custom", algorithm=request.algorithm),
                video_frequency=0,
                record_final_video=True,
            )
        )
    workspace = graph.create_root(
        config,
        branch=request.branch,
        label=request.label,
        prepare=True,
    )
    graph.save()
    return CreateRootResponse(
        node_id=workspace.id,
        branch=workspace.branch,
        status=workspace.status.value,
    )
