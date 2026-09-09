"""Tests for ``jarl.training.launch``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from jarl.training.config import RLRunConfig
from jarl.training.launch import (
    TRAIN_DETACH_ENV,
    TRAIN_WORKER_PID_NAME,
    build_create_and_train_command,
    build_resume_command,
    build_train_current_command,
    clear_train_worker_pid,
    live_training_pid,
    spawn_training,
    train_detach_enabled,
    write_run_config,
)


def test_write_run_config_roundtrip(tmp_path: Path) -> None:
    config = RLRunConfig()
    path = tmp_path / "config.json"

    write_run_config(config, path)

    loaded = path.read_text(encoding="utf-8")
    assert '"total_timesteps"' in loaded


def test_build_create_and_train_command_includes_label(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    config_path = tmp_path / "config.json"

    command = build_create_and_train_command(
        experiment_dir=exp_dir,
        config_path=config_path,
        python_executable="/usr/bin/python",
        label="smoke",
    )

    assert command[:3] == ["/usr/bin/python", "-m", "jarl.training.run"]
    assert "--create-root" in command
    assert command[command.index("--label") + 1] == "smoke"


def test_build_train_current_command_optional_node_id(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    config_path = tmp_path / "config.json"

    without_node = build_train_current_command(
        experiment_dir=exp_dir,
        config_path=config_path,
        python_executable="/usr/bin/python",
    )
    with_node = build_train_current_command(
        experiment_dir=exp_dir,
        config_path=config_path,
        python_executable="/usr/bin/python",
        node_id="node_a",
    )

    assert "--node-id" not in without_node
    assert with_node[with_node.index("--node-id") + 1] == "node_a"


def test_build_resume_command_requires_node_id(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    config_path = tmp_path / "config.json"

    command = build_resume_command(
        experiment_dir=exp_dir,
        config_path=config_path,
        node_id="node_a",
        python_executable="/usr/bin/python",
    )

    assert "--resume" in command
    assert command[command.index("--node-id") + 1] == "node_a"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("1", True, id="one"),
        pytest.param("true", True, id="true"),
        pytest.param("YES", True, id="yes"),
        pytest.param("0", False, id="zero"),
        pytest.param("", False, id="empty"),
    ],
)
def test_train_detach_enabled(raw: str, expected: bool, mocker: MockerFixture) -> None:
    mocker.patch.dict("os.environ", {TRAIN_DETACH_ENV: raw}, clear=False)

    assert train_detach_enabled() is expected


def test_spawn_training_uses_new_session_and_writes_pid(tmp_path: Path, mocker: MockerFixture) -> None:
    fake = mocker.Mock(pid=4242)
    fake.poll.return_value = None
    popen = mocker.patch("jarl.training.launch.subprocess.Popen", return_value=fake)
    mocker.patch.dict("os.environ", {TRAIN_DETACH_ENV: "1"}, clear=False)

    spawned = spawn_training(
        command=["python", "-m", "jarl.training.run"],
        experiment_dir=tmp_path,
        config_path=tmp_path / "config.json",
    )

    assert spawned.pid == 4242
    assert (tmp_path / TRAIN_WORKER_PID_NAME).read_text(encoding="utf-8").strip() == "4242"
    assert popen.call_args.kwargs["start_new_session"] is True
    assert TRAIN_DETACH_ENV not in popen.call_args.kwargs["env"]


def test_spawn_training_rejects_live_worker(tmp_path: Path, mocker: MockerFixture) -> None:
    (tmp_path / TRAIN_WORKER_PID_NAME).write_text("99\n", encoding="utf-8")
    mocker.patch("jarl.training.launch.os.kill")

    with pytest.raises(RuntimeError, match="still running"):
        spawn_training(
            command=["python"],
            experiment_dir=tmp_path,
            config_path=tmp_path / "config.json",
        )


def test_spawn_training_replaces_stale_pid(tmp_path: Path, mocker: MockerFixture) -> None:
    (tmp_path / TRAIN_WORKER_PID_NAME).write_text("99\n", encoding="utf-8")
    mocker.patch("jarl.training.launch.os.kill", side_effect=OSError)
    second = mocker.Mock(pid=100)
    second.poll.return_value = None
    mocker.patch("jarl.training.launch.subprocess.Popen", return_value=second)

    spawned = spawn_training(
        command=["python"],
        experiment_dir=tmp_path,
        config_path=tmp_path / "config.json",
    )

    assert spawned.pid == 100
    assert (tmp_path / TRAIN_WORKER_PID_NAME).read_text(encoding="utf-8").strip() == "100"


def test_live_training_pid_reaps_exited_handle_without_pidfile(tmp_path: Path, mocker: MockerFixture) -> None:
    worker = mocker.Mock(pid=77)
    worker.poll.return_value = None
    mocker.patch("jarl.training.launch.subprocess.Popen", return_value=worker)
    spawn_training(command=["python"], experiment_dir=tmp_path, config_path=tmp_path / "config.json")
    (tmp_path / TRAIN_WORKER_PID_NAME).unlink()
    worker.poll.return_value = 0

    assert live_training_pid(tmp_path) is None

    worker.wait.assert_called_once()
    assert not (tmp_path / TRAIN_WORKER_PID_NAME).exists()


def test_spawn_training_reaps_finished_worker_then_starts_another(tmp_path: Path, mocker: MockerFixture) -> None:
    first = mocker.Mock(pid=77)
    first.poll.return_value = None
    second = mocker.Mock(pid=88)
    second.poll.return_value = None
    mocker.patch("jarl.training.launch.subprocess.Popen", side_effect=[first, second])
    spawn_training(command=["python"], experiment_dir=tmp_path, config_path=tmp_path / "config.json")
    first.poll.return_value = 0

    spawned = spawn_training(command=["python"], experiment_dir=tmp_path, config_path=tmp_path / "config.json")

    assert spawned.pid == 88
    first.wait.assert_called_once()
    assert (tmp_path / TRAIN_WORKER_PID_NAME).read_text(encoding="utf-8").strip() == "88"


def test_clear_train_worker_pid_only_removes_matching_pid(tmp_path: Path) -> None:
    path = tmp_path / TRAIN_WORKER_PID_NAME
    path.write_text("123\n", encoding="utf-8")

    clear_train_worker_pid(tmp_path, pid=999)

    assert path.read_text(encoding="utf-8").strip() == "123"

    clear_train_worker_pid(tmp_path, pid=123)

    assert not path.exists()
