"""Tests for ExperimentGraph."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarl.config import BaseConfig, ConfigDiff
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.manifest import MANIFEST_FILENAME
from jarl.experiments.node import NodeWorkspace
from jarl.experiments.run_config import RunConfig, TrackingConfig

# ---- Test config ---- #


class SampleConfig(BaseConfig):
    """Concrete config for graph tests."""

    learning_rate: float = 0.1
    momentum: float = 0.9
    batch_size: int = 128


class CustomNodeWorkspace(NodeWorkspace):
    """Custom workspace subclass for reconstruction tests."""


class TrackingSampleConfig(RunConfig):
    """Run config with TensorBoard enabled for propagation tests."""

    learning_rate: float = 0.1


# ---- Fixtures ---- #


@pytest.fixture()
def exp_dir(tmp_path: Path) -> Path:
    """Temp experiment directory."""
    return tmp_path / "runs" / "test_exp"


@pytest.fixture()
def graph(exp_dir: Path) -> ExperimentGraph[SampleConfig]:
    """An experiment graph with a root node."""
    g: ExperimentGraph[SampleConfig] = ExperimentGraph(exp_dir)
    g.create_root(config=SampleConfig(), branch="main", label="baseline")
    return g


# ---- Root creation ---- #


class TestCreateRoot:
    """Tests for root node creation."""

    def test_creates_root_node(self, exp_dir: Path) -> None:
        g = ExperimentGraph(exp_dir)

        root = g.create_root(config=SampleConfig(), branch="main", label="baseline")

        assert root.branch == "main"
        assert root.node_metadata.label == "baseline"
        assert root.node_metadata.parent_id is None
        assert g.head("main") is root
        assert g.current_node is root
        assert root.id.startswith("main_baseline_")

    @pytest.mark.parametrize(
        ("filename",),
        [
            pytest.param("config.json", id="config"),
            pytest.param("run_metadata.json", id="run_metadata"),
            pytest.param(MANIFEST_FILENAME, id="manifest"),
        ],
    )
    def test_persists_experiment_manifest_files(self, exp_dir: Path, filename: str) -> None:
        g = ExperimentGraph(exp_dir)

        g.create_root(config=SampleConfig())

        assert (exp_dir / filename).is_file()

    def test_root_has_no_parent(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        assert root.node_metadata.parent_id is None

    def test_create_node_persists_resolved_config_snapshot(self, exp_dir: Path) -> None:
        g = ExperimentGraph(exp_dir)
        root = g.create_root(config=SampleConfig(learning_rate=0.1), branch="main")

        node_config = json.loads(root.config_path.read_text(encoding="utf-8"))

        assert root.config_path.is_file()
        assert node_config["learning_rate"] == 0.1

    def test_extend_persists_child_resolved_config(self, graph: ExperimentGraph[SampleConfig]) -> None:
        child = graph.extend("main", config=SampleConfig(learning_rate=0.01))

        node_config = json.loads(child.config_path.read_text(encoding="utf-8"))

        assert node_config["learning_rate"] == 0.01

    def test_duplicate_root_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(RuntimeError, match="already has a root"):
            graph.create_root(config=SampleConfig())

    def test_creates_branch_pointer(self, graph: ExperimentGraph[SampleConfig]) -> None:
        assert graph.branch_heads["main"] is graph.head("main")

    def test_create_root_on_nonempty_dir_raises(self, exp_dir: Path) -> None:
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_metadata.json").write_text("{}")

        graph = ExperimentGraph[SampleConfig](exp_dir)

        with pytest.raises(RuntimeError, match="not empty"):
            graph.create_root(config=SampleConfig())

    def test_node_dir_created(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        assert root.path.exists()
        assert (root.path / "node.json").exists()


# ---- Public experiment API ---- #


class TestPublicExperimentApi:
    """Tests for the public experiment navigation and inspection API."""

    def test_branch_heads_and_get_branches_match(self, graph: ExperimentGraph[SampleConfig]) -> None:
        assert graph.get_branches() == graph.branch_heads
        assert graph.branch_heads["main"].id == graph.head("main").id

    def test_get_node_lookup(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        assert graph.get_node(root.id) is root

    def test_current_node_updates_after_extend(self, graph: ExperimentGraph[SampleConfig]) -> None:
        extended = graph.extend(label="step2")

        assert graph.current_node is extended

    def test_execution_state_is_persisted(self, graph: ExperimentGraph[SampleConfig], exp_dir: Path) -> None:
        assert graph.execution_state.status == "idle"

        manifest = json.loads((exp_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

        assert manifest["execution_state"]["status"] == "idle"
        assert manifest["current_node"] == graph.current_node.id

    def test_is_branch_head_on_root_and_extended_node(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        extended = graph.extend(label="step2")

        assert graph.is_branch_head(root.id) is False
        assert graph.is_branch_head(root) is False
        assert graph.is_branch_head(extended.id) is True
        assert graph.is_branch_head(extended) is True

    def test_require_branch_head_raises_for_non_head(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        graph.extend(label="step2")

        with pytest.raises(ValueError, match="not the head"):
            graph.require_branch_head(root.id)
        with pytest.raises(ValueError, match="not the head"):
            graph.require_branch_head(root)

    def test_require_branch_head_returns_workspace(self, graph: ExperimentGraph[SampleConfig]) -> None:
        extended = graph.extend(label="step2")

        assert graph.require_branch_head(extended) is graph.get_node(extended.id)


# ---- Fork ---- #


class TestFork:
    """Tests for branching from a node."""

    def test_fork_creates_new_branch(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        forked = graph.fork("lr_exp", from_node=root, config=SampleConfig(learning_rate=0.01), label="higher_lr")

        assert forked.branch == "lr_exp"
        assert graph.branch_heads["lr_exp"] is forked

    def test_fork_defaults_to_current_node(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        forked = graph.fork("lr_exp", config=SampleConfig(learning_rate=0.01))

        assert forked.node_metadata.parent_id == root.id
        assert graph.current_node is forked

    def test_fork_stores_config_overrides(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        forked = graph.fork("lr_exp", from_node=root, config=SampleConfig(learning_rate=0.01))

        assert forked.node_metadata.config_overrides == {"learning_rate": 0.01}

    def test_fork_parent_is_source_node(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        forked = graph.fork("lr_exp", from_node=root)

        assert forked.node_metadata.parent_id == root.id

    def test_fork_by_id_string(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        forked = graph.fork("lr_exp", from_node=root.id)

        assert forked.node_metadata.parent_id == root.id

    def test_duplicate_branch_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        with pytest.raises(ValueError, match="already exists"):
            graph.fork("main", from_node=root)

    def test_fork_unknown_node_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(KeyError, match="not found"):
            graph.fork("new_branch", from_node="nonexistent_node")

    def test_fork_with_label_and_description(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        forked = graph.fork(
            "exp_a",
            from_node=root,
            label="test_fork",
            description="Testing fork with metadata",
            metadata={"author": "test"},
        )

        assert forked.node_metadata.label == "test_fork"
        assert forked.node_metadata.description == "Testing fork with metadata"
        assert forked.node_metadata.metadata == {"author": "test"}


# ---- Extend ---- #


class TestExtend:
    """Tests for extending a branch with a new node."""

    def test_extend_branch_head_adds_node(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        extended = graph.extend("main", label="epoch_10")

        assert extended.branch == "main"
        assert extended.node_metadata.parent_id == root.id
        assert graph.head("main").id == extended.id

    def test_extend_defaults_to_current_node(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        graph.checkout(root)

        extended = graph.extend(label="from_current")

        assert extended.node_metadata.parent_id == root.id
        assert graph.current_node is extended

    def test_extend_from_checked_out_ancestor(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        tip = graph.extend("main", label="step2")

        graph.checkout(root)
        extended = graph.extend(label="branch_from_root")

        assert extended.node_metadata.parent_id == root.id
        assert graph.head("main").id == extended.id
        assert tip.id in graph.all_nodes
        assert graph.current_node is extended

    def test_extend_unknown_branch_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(KeyError, match="not found"):
            graph.extend("nonexistent")


# ---- Checkout ---- #


class TestCheckout:
    """Tests for navigating the tree via checkout."""

    def test_checkout_moves_current_node_only(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        tip = graph.extend("main", label="step2")

        graph.checkout(root)

        assert graph.current_node is root
        assert graph.head("main").id == tip.id

    def test_checkout_keeps_abandoned_nodes(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        extended = graph.extend("main", label="step2")

        graph.checkout(root)

        assert extended.id in graph.all_nodes

    def test_checkout_unknown_node_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(KeyError, match="not found"):
            graph.checkout("nonexistent")

    def test_checkout_persists_current_node(self, graph: ExperimentGraph[SampleConfig], exp_dir: Path) -> None:
        root = graph.head("main")
        graph.extend("main", label="step2")

        graph.checkout(root)

        manifest = json.loads((exp_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["current_node"] == root.id


# ---- Lineage ---- #


class TestLineage:
    """Tests for lineage traversal."""

    def test_root_lineage_is_single_node(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        lineage = graph.get_lineage(root)

        assert len(lineage) == 1
        assert lineage[0].id == root.id

    def test_extended_lineage_in_order(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        step2 = graph.extend("main", label="step2")
        step3 = graph.extend("main", label="step3")

        lineage = graph.get_lineage(step3)

        assert [ws.id for ws in lineage] == [root.id, step2.id, step3.id]

    def test_forked_lineage_includes_parent(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        forked = graph.fork("fork_a", from_node=root)

        lineage = graph.get_lineage(forked)

        assert len(lineage) == 2
        assert lineage[0].id == root.id
        assert lineage[1].id == forked.id

    def test_unknown_node_raises(self, graph: ExperimentGraph[SampleConfig]) -> None:
        with pytest.raises(KeyError, match="not found"):
            graph.get_lineage("nonexistent")


# ---- Config resolution ---- #


class TestResolveConfig:
    """Tests for config resolution along lineage."""

    def test_root_resolves_to_base(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        config = graph.resolve_config(root)

        assert config.learning_rate == 0.1
        assert config.batch_size == 128

    def test_fork_applies_overrides(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        graph.fork("lr_exp", from_node=root, config=SampleConfig(learning_rate=0.01))

        config = graph.resolve_config(graph.head("lr_exp"))

        assert config.learning_rate == 0.01
        assert config.batch_size == 128

    def test_chained_overrides(self, graph: ExperimentGraph[SampleConfig]) -> None:
        graph.extend("main", config=SampleConfig(learning_rate=0.05))
        graph.extend("main", config=SampleConfig(learning_rate=0.05, batch_size=256))

        config = graph.resolve_config(graph.head("main"))

        assert config.learning_rate == 0.05
        assert config.batch_size == 256
        assert config.momentum == 0.9

    def test_config_diff_between_nodes(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        forked = graph.fork("lr_exp", from_node=root, config=SampleConfig(learning_rate=0.01))

        diff = graph.get_config_diff(root, forked)

        assert diff == ConfigDiff(added={}, removed={}, changed={"learning_rate": (0.1, 0.01)})


# ---- Metrics along lineage ---- #


class TestMetricsAlongLineage:
    """Tests for collecting metrics along a lineage."""

    def test_empty_metrics(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        metrics = graph.get_metrics_along_lineage(root)

        assert metrics == [{}]

    def test_metrics_from_saved_nodes(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")
        root.save_metrics({"loss": 0.5})
        step2 = graph.extend("main")
        step2.save_metrics({"loss": 0.3})

        metrics = graph.get_metrics_along_lineage(step2)

        assert len(metrics) == 2
        assert metrics[0]["loss"] == 0.5
        assert metrics[1]["loss"] == 0.3


# ---- Immutability ---- #


class TestCompletedNodeImmutability:
    """Tests for completed-node training guards."""

    def test_completed_node_cannot_reenter(self, graph: ExperimentGraph[SampleConfig]) -> None:
        root = graph.head("main")

        with root:
            pass

        with pytest.raises(RuntimeError, match="Cannot re-enter a completed node"), root:
            pass


# ---- Persistence & reconstruction ---- #


class TestPersistence:
    """Tests for experiment save/load and from_directory reconstruction."""

    def test_from_directory_restores_nodes(self, graph: ExperimentGraph[SampleConfig], exp_dir: Path) -> None:
        root = graph.head("main")
        graph.fork("side", from_node=root, config=SampleConfig(learning_rate=0.01))

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)

        assert len(restored.all_nodes) == 2
        assert "main" in restored.branch_heads
        assert "side" in restored.branch_heads

    def test_from_directory_preserves_lineage(self, graph: ExperimentGraph[SampleConfig], exp_dir: Path) -> None:
        graph.head("main")
        graph.extend("main", label="step2")

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)

        lineage = restored.get_lineage(restored.head("main"))
        assert len(lineage) == 2

    def test_from_directory_resolves_config(self, graph: ExperimentGraph[SampleConfig], exp_dir: Path) -> None:
        root = graph.head("main")
        graph.fork("lr_exp", from_node=root, config=SampleConfig(learning_rate=0.01))

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)

        config = restored.resolve_config(restored.head("lr_exp"))
        assert config.learning_rate == 0.01

    def test_from_directory_restores_current_node(self, graph: ExperimentGraph[SampleConfig], exp_dir: Path) -> None:
        root = graph.head("main")
        graph.extend("main", label="step2")
        graph.checkout(root)

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)

        assert restored.current_node.id == root.id

    def test_from_directory_restores_custom_workspace(self, exp_dir: Path) -> None:
        graph = ExperimentGraph[SampleConfig](exp_dir)
        graph.create_root(config=SampleConfig(), branch="alt", workspace_cls=CustomNodeWorkspace)

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)

        assert isinstance(restored.head("alt"), CustomNodeWorkspace)

    def test_from_directory_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ExperimentGraph.from_directory(tmp_path / "nonexistent")

    def test_from_directory_invalid_manifest_raises(self, exp_dir: Path) -> None:
        graph = ExperimentGraph[SampleConfig](exp_dir)
        graph.create_root(config=SampleConfig())

        manifest_path = exp_dir / MANIFEST_FILENAME
        manifest_path.write_text('{"version": 1, "nodes": [], "edges": [], "branch_heads": {"main": "missing"}}')

        with pytest.raises(ValueError, match="unknown node"):
            ExperimentGraph.from_directory(exp_dir, config_cls=SampleConfig)

    def test_create_root_propagates_tracking_config(self, exp_dir: Path) -> None:
        graph: ExperimentGraph[TrackingSampleConfig] = ExperimentGraph(
            exp_dir,
            base_config=TrackingSampleConfig(tracking=TrackingConfig(track_tensorboard=True)),
        )
        root = graph.create_root(
            config=TrackingSampleConfig(tracking=TrackingConfig(track_tensorboard=True)),
            branch="main",
        )

        assert root.tracking.track_tensorboard is True

        with root:
            root.log_scalar(1, "loss", 0.5)

        assert list(root.tensorboard_dir.glob("events.out.tfevents.*"))

    def test_from_directory_restores_tracking_config(self, exp_dir: Path) -> None:
        graph: ExperimentGraph[TrackingSampleConfig] = ExperimentGraph(
            exp_dir,
            base_config=TrackingSampleConfig(tracking=TrackingConfig(track_tensorboard=True)),
        )
        root = graph.create_root(
            config=TrackingSampleConfig(tracking=TrackingConfig(track_tensorboard=True)),
            branch="main",
        )
        with root:
            root.log_scalar(1, "accuracy", 0.8)

        restored = ExperimentGraph.from_directory(exp_dir, config_cls=TrackingSampleConfig)
        restored_root = restored.head("main")
        assert restored_root.tracking.track_tensorboard is True
        assert list(restored_root.tensorboard_dir.glob("events.out.tfevents.*"))

    def test_repr(self, graph: ExperimentGraph[SampleConfig]) -> None:
        result = repr(graph)

        assert "nodes=1" in result
        assert "branches=1" in result
