"""Base contracts for materializing reusable experiment cases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from jarl.config import BaseConfig
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeWorkspace

__all__ = [
    "CaseDefinition",
    "ExperimentCase",
    "ExperimentCaseContext",
]


class CaseDefinition(Protocol):
    """Minimal contract consumed by case registries."""

    @property
    def spec(self) -> CaseSpec:
        """Return stable case metadata."""
        ...


@dataclass(slots=True)
class ExperimentCaseContext[ConfigT: BaseConfig]:
    """Materialized experiment and stable logical node aliases.

    Args:
        experiment_dir: Root directory containing the experiment.
        config_cls: Configuration model used to reload its graph.
    """

    experiment_dir: Path
    config_cls: type[ConfigT]
    _aliases: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    @property
    def aliases(self) -> MappingProxyType[str, str]:
        """Return a read-only view of logical aliases to node ids."""
        return MappingProxyType(self._aliases)

    def register_alias(self, alias: str, node: str | NodeWorkspace) -> None:
        """Associate a stable logical alias with a materialized node.

        Args:
            alias: Logical name used by cases and validators.
            node: Node id or workspace created during materialization.

        Raises:
            ValueError: If the alias is empty or already registered.
        """
        normalized = alias.strip()
        if not normalized:
            raise ValueError("Node alias must not be empty.")
        if normalized in self._aliases:
            msg = f"Node alias already registered: {normalized!r}"
            raise ValueError(msg)
        self._aliases[normalized] = node.id if isinstance(node, NodeWorkspace) else node

    def node_id(self, alias: str) -> str:
        """Resolve a logical node alias.

        Args:
            alias: Previously registered logical name.

        Returns:
            Materialized node id.

        Raises:
            KeyError: If the alias is unknown.
        """
        return self._aliases[alias]

    def reload_graph(self) -> ExperimentGraph[ConfigT]:
        """Reload and return the materialized experiment graph."""
        return ExperimentGraph.from_directory(self.experiment_dir, config_cls=self.config_cls)


class ExperimentCase[ConfigT: BaseConfig](ABC):
    """Reusable instructions for creating an experiment graph scenario."""

    def __init__(self, spec: CaseSpec, config_cls: type[ConfigT]) -> None:
        """Initialize a case from metadata and its graph config model."""
        self._spec = spec
        self._config_cls = config_cls

    @property
    def spec(self) -> CaseSpec:
        """Return stable case metadata."""
        return self._spec

    @property
    def config_cls(self) -> type[ConfigT]:
        """Return the configuration model used by this case."""
        return self._config_cls

    def materialize(self, destination: str | Path) -> ExperimentCaseContext[ConfigT]:
        """Create the case in an empty destination directory.

        Args:
            destination: New or empty directory for the experiment.

        Returns:
            Context containing the experiment path and logical node aliases.

        Raises:
            ValueError: If the destination exists and is not an empty directory.
        """
        experiment_dir = Path(destination).resolve()
        if experiment_dir.exists():
            if not experiment_dir.is_dir() or any(experiment_dir.iterdir()):
                msg = f"Experiment case destination must be an empty directory: {experiment_dir}"
                raise ValueError(msg)
        else:
            experiment_dir.mkdir(parents=True)

        context = ExperimentCaseContext(
            experiment_dir=experiment_dir,
            config_cls=self._config_cls,
        )
        self.build(context)
        context.reload_graph()
        return context

    @abstractmethod
    def build(self, context: ExperimentCaseContext[ConfigT]) -> None:
        """Populate an empty case context using public experiment APIs."""
