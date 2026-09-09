"""Versioned experiment tree manifest and execution state."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from jarl.metadata import now_iso

MANIFEST_VERSION: Final[int] = 1
"""Supported experiment manifest schema version."""

MANIFEST_FILENAME = "experiment.json"
"""Primary manifest filename for experiment trees."""


class ExecutionState(BaseModel):
    """Execution snapshot persisted with the experiment manifest.

    Args:
        updated_at: ISO-8601 timestamp of the last execution-state change.
        status: Coarse experiment execution status.
    """

    model_config = ConfigDict(frozen=True)

    updated_at: str = Field(default_factory=now_iso, description="Last update timestamp.")
    status: Literal["idle", "active", "interrupted"] = Field(
        default="idle",
        description="Coarse execution status for the experiment session.",
    )


class ExperimentManifest(BaseModel):
    """Versioned on-disk representation of an experiment tree.

    Args:
        version: Manifest schema version.
        nodes: Node identifiers present in the tree.
        edges: Directed parent-child edges as `(parent_id, child_id)` pairs.
        branch_heads: Mapping from branch name to the head node id.
        current_node: Node id selected for navigation and default fork/extend.
        execution_state: Persisted execution snapshot for the experiment.
    """

    model_config = ConfigDict(frozen=True)

    version: Final[int] = MANIFEST_VERSION
    nodes: list[str] = Field(default_factory=list, description="Node ids in the experiment tree.")
    edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Parent-child edges as (parent_id, child_id) pairs.",
    )
    branch_heads: dict[str, str] = Field(default_factory=dict, description="Branch name to head node id.")
    current_node: str = Field(default="", description="Currently selected node id.")
    execution_state: ExecutionState = Field(default_factory=ExecutionState)

    def validate_tree(self) -> None:
        """Validate tree invariants for nodes, edges and branch pointers.

        Raises:
            ValueError: If the manifest describes an invalid tree structure.
        """
        node_set = set(self.nodes)
        if len(node_set) != len(self.nodes):
            msg = "Duplicate node ids in manifest."
            raise ValueError(msg)

        child_to_parents: dict[str, list[str]] = {}
        for parent_id, child_id in self.edges:
            if parent_id not in node_set or child_id not in node_set:
                msg = f"Edge references unknown node: ({parent_id}, {child_id})"
                raise ValueError(msg)
            child_to_parents.setdefault(child_id, []).append(parent_id)

        for child_id, parents in child_to_parents.items():
            if len(parents) > 1:
                msg = f"Node {child_id} has multiple parents; tree structure required."
                raise ValueError(msg)

        for branch, head_id in self.branch_heads.items():
            if head_id not in node_set:
                msg = f"Branch head {branch!r} references unknown node {head_id!r}."
                raise ValueError(msg)

        if self.current_node and self.current_node not in node_set:
            msg = f"Current node {self.current_node!r} is not present in manifest nodes."
            raise ValueError(msg)
