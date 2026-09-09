"""Built-in empty experiment case for agent setup scenarios."""

from __future__ import annotations

from jarl.experiments.cases.case import ExperimentCase, ExperimentCaseContext
from jarl.experiments.cases.spec import CaseSpec
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.manifest import ExperimentManifest
from jarl.training.config import RLRunConfig
from jarl.utils.io import write_text_atomic

__all__ = ["CASE", "EmptyExperimentCase"]


class EmptyExperimentCase(ExperimentCase[RLRunConfig]):
    """Materialize a valid experiment manifest without graph nodes."""

    def __init__(self) -> None:
        """Initialize the built-in empty experiment case."""
        super().__init__(
            CaseSpec(
                id="empty_experiment",
                title="Empty experiment",
                description="Valid RL experiment scaffold with no graph nodes.",
                tags=frozenset({"empty", "setup"}),
            ),
            RLRunConfig,
        )

    def build(self, context: ExperimentCaseContext[RLRunConfig]) -> None:
        """Write base configuration and an empty experiment manifest."""
        config = RLRunConfig()
        graph = ExperimentGraph(context.experiment_dir, base_config=config)
        config.save(graph.layout.config_path)
        write_text_atomic(
            graph.layout.manifest_path,
            ExperimentManifest().model_dump_json(indent=2),
        )


CASE = EmptyExperimentCase()
"""Default empty experiment case instance."""
