"""Tests for compact dict projections on operation responses."""

from __future__ import annotations

from jarl.experiments.graph import ExperimentGraph
from jarl.operations.graph.create_root import CreateRootRequest, create_root
from jarl.operations.graph.summary import SummaryRequest, summary
from jarl.training.config import RLRunConfig


def test_to_compact_dict_omits_none_fields(empty_graph: ExperimentGraph[RLRunConfig]) -> None:
    response = create_root(empty_graph, CreateRootRequest(label="baseline"))

    compact = response.to_compact_dict()

    assert compact == {
        "node_id": response.node_id,
        "branch": "main",
        "status": "prepared",
    }
    assert "None" not in str(compact)


def test_summary_compact_dict_is_json_friendly(prepared_root: ExperimentGraph[RLRunConfig]) -> None:
    compact = summary(prepared_root, SummaryRequest()).to_compact_dict()

    assert isinstance(compact["nodes"], list)
    assert "config_highlights" in compact["nodes"][0]
