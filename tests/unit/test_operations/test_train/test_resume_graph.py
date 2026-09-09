"""Tests for resume mutating the injected experiment graph."""

from __future__ import annotations

import pytest

from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.node import NodeStatus, NodeWorkspace
from jarl.operations.train.resume import ResumeRequest, resume
from jarl.operations.train.run import RunRequest, run
from jarl.training.config import RLRunConfig
from jarl.training.schedule import TrainingSchedule
from jarl.training.trainer import EnvFactory
from tests.helpers.fake_trainer import fake_trainer


class TestResumeGraphMutation:
    def test_resume_updates_injected_graph_and_disk(
        self,
        prepared_root: ExperimentGraph[RLRunConfig],
    ) -> None:
        node_id = prepared_root.current_node.id

        def failing_trainer(
            _workspace: NodeWorkspace,
            config: RLRunConfig,
            schedule: TrainingSchedule,
            *,
            env_factory: EnvFactory | None = None,
        ) -> None:
            del config, schedule, env_factory
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            run(prepared_root, RunRequest(), trainer=failing_trainer)

        assert prepared_root.get_node(node_id).status == NodeStatus.FAILED

        response = resume(prepared_root, ResumeRequest(node_id=node_id), trainer=fake_trainer)

        assert response.status == NodeStatus.COMPLETED.value
        assert prepared_root.get_node(node_id).status == NodeStatus.COMPLETED
        assert prepared_root.current_node.id == node_id
        assert prepared_root.get_node(node_id).metrics_jsonl_path.is_file()

        reloaded = ExperimentGraph.from_directory(prepared_root.layout.root, config_cls=RLRunConfig)
        assert reloaded.get_node(node_id).status == NodeStatus.COMPLETED
