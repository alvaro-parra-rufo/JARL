"""Tests for Runner Lab checkout widget sync."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pytest_mock import MockerFixture

from jarl.app.lib.session import (
    NODE_PICKER_WIDGET_KEYS,
    PENDING_NODE_PICKER_SYNC_KEY,
    ActiveAgenticCaseRun,
    active_agentic_case_run,
    apply_pending_node_picker_sync,
    checkout_experiment_node,
    poll_active_agentic_case_run,
    request_node_picker_sync,
    set_active_agentic_case_run,
)


class _FakeSessionState(dict):
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        if key in self:
            return super().pop(key)
        return default


def test_apply_pending_node_picker_sync_updates_all_keys(mocker: MockerFixture) -> None:
    session = _FakeSessionState({"entrenar_node": "old_node"})
    mocker.patch("streamlit.session_state", session, create=True)

    request_node_picker_sync("new_node")
    applied = apply_pending_node_picker_sync()

    assert applied == "new_node"
    for key in NODE_PICKER_WIDGET_KEYS:
        assert session[key] == "new_node"
    assert PENDING_NODE_PICKER_SYNC_KEY not in session


def test_checkout_experiment_node_persists_and_queues_widget_sync(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    from jarl.experiments.graph import ExperimentGraph
    from jarl.training.config import RLRunConfig

    exp_dir = tmp_path / "exp"
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    root = graph.create_root(RLRunConfig(), label="root", prepare=True)
    child = graph.fork("alt", from_node=root, label="child", prepare=True)
    graph.save()

    session = _FakeSessionState({"entrenar_node": root.id})
    mocker.patch("streamlit.session_state", session, create=True)
    bump = mocker.patch("jarl.app.lib.session.bump_graph_revision")

    checkout_experiment_node(exp_dir, child.id)

    reloaded = ExperimentGraph.from_directory(exp_dir, config_cls=RLRunConfig)
    assert reloaded.current_node.id == child.id
    assert session[PENDING_NODE_PICKER_SYNC_KEY] == child.id
    bump.assert_called_once()

    apply_pending_node_picker_sync()
    assert session["entrenar_node"] == child.id


def test_poll_agentic_case_run_clears_finished_process(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    session = _FakeSessionState()
    mocker.patch("streamlit.session_state", session, create=True)
    process = mocker.Mock()
    process.poll.return_value = 1
    run = ActiveAgenticCaseRun(
        process=process,
        log_path=tmp_path / "case.log",
        case_id="case",
        experiment_dir=tmp_path / "experiment",
    )
    bump = mocker.patch("jarl.app.lib.session.bump_graph_revision")
    set_active_agentic_case_run(run)

    exit_code = poll_active_agentic_case_run()

    assert exit_code == 1
    assert active_agentic_case_run() is None
    bump.assert_not_called()
