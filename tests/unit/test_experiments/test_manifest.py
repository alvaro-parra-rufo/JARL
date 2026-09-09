"""Tests for experiment manifest schema and validation."""

from __future__ import annotations

import pytest

from jarl.experiments.manifest import ExecutionState, ExperimentManifest


class TestExperimentManifestValidation:
    """Tests for manifest tree invariants."""

    def test_valid_tree_passes(self) -> None:
        manifest = ExperimentManifest(
            nodes=["root", "child"],
            edges=[("root", "child")],
            branch_heads={"main": "child"},
            current_node="child",
            execution_state=ExecutionState(status="idle"),
        )

        manifest.validate_tree()

    def test_duplicate_parent_raises(self) -> None:
        manifest = ExperimentManifest(
            nodes=["root", "a", "b"],
            edges=[("root", "a"), ("root", "b"), ("a", "b")],
            branch_heads={"main": "b"},
            current_node="b",
        )

        with pytest.raises(ValueError, match="multiple parents"):
            manifest.validate_tree()

    def test_unknown_branch_head_raises(self) -> None:
        manifest = ExperimentManifest(
            nodes=["root"],
            edges=[],
            branch_heads={"main": "missing"},
            current_node="root",
        )

        with pytest.raises(ValueError, match="unknown node"):
            manifest.validate_tree()

    def test_unknown_current_node_raises(self) -> None:
        manifest = ExperimentManifest(
            nodes=["root"],
            edges=[],
            branch_heads={"main": "root"},
            current_node="missing",
        )

        with pytest.raises(ValueError, match="not present"):
            manifest.validate_tree()
