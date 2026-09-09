"""Shared fixtures for functional ``jarl.agentic`` tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from jarl.agentic.cases import BaseAgenticCase
from jarl.experiments.cases.builtin.navix_ppo_rollout_long_trained_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP as LONG_TRAINED_ROLLOUT_CHECKPOINT_STEP,
)
from jarl.experiments.cases.builtin.navix_ppo_rollout_trained_checkpoint import (
    ROLLOUT_CHECKPOINT_STEP as TRAINED_ROLLOUT_CHECKPOINT_STEP,
)
from jarl.experiments.cases.builtin.navix_ppo_rollout_trained_checkpoint import (
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SEED,
)
from jarl.experiments.graph import ExperimentGraph
from jarl.experiments.manifest import ExperimentManifest
from jarl.training.config import RLRunConfig
from jarl.utils.io import write_text_atomic
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel
from tests.helpers.fork_from_latest_not_best_fake import ForkFromLatestNotBestFake
from tests.helpers.metrics_analysis_fake import MetricsAnalysisFake


@pytest.fixture()
def empty_experiment(tmp_path: Path) -> Path:
    """Experiment directory with manifest and base config but no nodes."""
    exp_dir = tmp_path / "exp"
    config = RLRunConfig()
    graph = ExperimentGraph(exp_dir, base_config=config)
    config.save(graph.layout.config_path)
    write_text_atomic(
        graph.layout.manifest_path,
        ExperimentManifest().model_dump_json(indent=2),
    )
    return exp_dir


@pytest.fixture()
def demo_tree(tmp_path: Path) -> Path:
    """Prepared root with a two-node ``exp`` branch for read and mutation scenarios."""
    exp_dir = tmp_path / "demo"
    graph = ExperimentGraph(exp_dir, base_config=RLRunConfig())
    graph.create_root(RLRunConfig(), branch="main", label="root", prepare=True)
    head = graph.current_node
    graph.fork("exp", from_node=head, label="first", prepare=True)
    graph.extend("exp", label="second", prepare=True)
    graph.save()
    return exp_dir


@pytest.fixture()
def agentic_case_llm(case: BaseAgenticCase) -> ToolBindingFakeChatModel:
    """Return scripted tool calls for one parametrized built-in case."""
    if case.spec.id == "graph_fork_from_latest_not_best":
        return ForkFromLatestNotBestFake(messages=iter(()))
    if case.spec.id == "subagent_metrics_analysis_improving":
        return MetricsAnalysisFake(
            messages=iter(()),
            evidence_ids=[
                "eval_return.window_initial.mean",
                "eval_return.window_final.mean",
            ],
        )
    scripts: dict[str, tuple[AIMessage, ...]] = {
        "graph_create_root_baseline": (
            _tool_call("graph_create_root", {"label": "baseline"}, "create_root"),
            AIMessage(content="Setup complete."),
            *_finish("Experiment ready."),
        ),
        "graph_extend_same_branch": (
            _tool_call(
                "graph_extend",
                {"branch": "main", "label": "continued", "prepare": True},
                "extend",
            ),
            *_finish("Branch extended."),
        ),
        "graph_fork_parallel_branch": (
            _tool_call(
                "graph_fork",
                {"branch": "exploration", "label": "parallel", "prepare": True},
                "fork",
            ),
            *_finish("Parallel branch created."),
        ),
        "graph_fork_ppo_overrides": (
            _tool_call(
                "graph_fork",
                {
                    "branch": "ppo_tuned",
                    "label": "tuned_fork",
                    "prepare": True,
                    "config_overrides": {
                        "algorithm.learning_rate": 0.001,
                        "algorithm.gamma": 0.97,
                        "algorithm.entropy_coef": 0.08,
                        "algorithm.gae_lambda": 0.9,
                        "algorithm.clip_range": 0.15,
                    },
                },
                "fork_ppo",
            ),
            *_finish("Prepared tuned fork."),
        ),
        "graph_read_without_mutation": (
            _tool_call("graph_summary", {}, "summary"),
            *_finish("The experiment has three nodes."),
        ),
        "train_recovery_status_resume": (
            _tool_call("train_recovery_status", {}, "recovery"),
            *_finish("Se puede reanudar con recommended_action=train_resume."),
        ),
        "env_navix_maps_read": (
            _tool_call("env_navix_maps", {"categories": ["door_key"]}, "navix_maps"),
            *_finish("DoorKey maps from the catalog include Navix-DoorKey-5x5-v0 and Navix-DoorKey-8x8-v0."),
        ),
        "env_navix_maps_search_key": (
            _tool_call(
                "env_navix_maps",
                {"categories": ["door_key", "key_corridor"]},
                "navix_maps_key",
            ),
            *_finish("Maps that use a key include Navix-DoorKey-5x5-v0 and Navix-KeyCorridorS3R1-v0."),
        ),
        "env_navix_maps_query_lava": (
            _tool_call("env_navix_maps", {"query": "lava"}, "navix_maps_lava"),
            *_finish("Lava maps include Navix-LavaGap-S5-v0."),
        ),
        "env_navix_maps_difficulty_easy": (
            _tool_call(
                "env_navix_maps",
                {"difficulty_max": 20},
                "navix_maps_easy",
            ),
            *_finish("Easy maps include Navix-Empty-5x5-v0."),
        ),
        "env_navix_maps_transfer_ready": (
            _tool_call(
                "env_navix_maps",
                {"jarl_transfer_ready_only": True},
                "navix_maps_transfer",
            ),
            *_finish("Transfer-ready maps include Navix-Empty-5x5-v0."),
        ),
        "env_navix_maps_filters_combo": (
            _tool_call(
                "env_navix_maps",
                {
                    "categories": ["lava_gap"],
                    "query": "lava",
                    "difficulty_min": 40,
                    "difficulty_max": 60,
                    "jarl_transfer_ready_only": True,
                },
                "navix_maps_combo",
            ),
            *_finish("Matching lava maps include Navix-LavaGap-S5-v0."),
        ),
        "graph_extend_recover_override": (
            _tool_call(
                "graph_extend",
                {
                    "branch": "main",
                    "label": "recovered",
                    "prepare": True,
                    "config_overrides": {"algorithm.eval_frequency": 128},
                },
                "invalid_extend",
            ),
            _tool_call(
                "graph_extend",
                {
                    "branch": "main",
                    "label": "recovered",
                    "prepare": True,
                    "config_overrides": {
                        "algorithm.evaluation_and_save_frequency": 128,
                    },
                },
                "recovered_extend",
            ),
            *_finish("Override corrected."),
        ),
        "graph_extend_ppo_overrides": (
            _tool_call(
                "graph_extend",
                {
                    "branch": "main",
                    "label": "tuned",
                    "prepare": True,
                    "config_overrides": {
                        "algorithm.learning_rate": 0.001,
                        "algorithm.gamma": 0.97,
                        "algorithm.entropy_coef": 0.08,
                        "algorithm.gae_lambda": 0.9,
                        "algorithm.clip_range": 0.15,
                    },
                },
                "extend_ppo",
            ),
            *_finish("Prepared tuned child on main."),
        ),
        "graph_checkpoint_rollout_analysis": (
            _tool_call(
                "graph_checkpoint_rollout",
                {
                    "checkpoint_step": TRAINED_ROLLOUT_CHECKPOINT_STEP,
                    "seed": ROLLOUT_SEED,
                    "max_steps": ROLLOUT_MAX_STEPS,
                },
                "checkpoint_rollout_trained",
            ),
            *_finish("Rollout analizado."),
        ),
        "graph_checkpoint_rollout_analysis_long": (
            _tool_call(
                "graph_checkpoint_rollout",
                {
                    "checkpoint_step": LONG_TRAINED_ROLLOUT_CHECKPOINT_STEP,
                    "seed": ROLLOUT_SEED,
                    "max_steps": ROLLOUT_MAX_STEPS,
                },
                "checkpoint_rollout_long",
            ),
            *_finish("Rollout largo analizado."),
        ),
        "navix_empty_variant_spec": (
            _tool_call("graph_reward", {}, "reward"),
            _tool_call("graph_set_reward", {"goal_approach": 2.0}, "set_reward"),
            *_finish("Configurable mix updated."),
        ),
    }
    try:
        messages = scripts[case.spec.id]
    except KeyError as exc:
        msg = f"No scripted LLM responses for agentic case {case.spec.id!r}."
        raise ValueError(msg) from exc
    return ToolBindingFakeChatModel(messages=iter(messages))


def _finish(content: str, call_id: str = "session_finish") -> tuple[AIMessage, AIMessage]:
    """Append a scripted `session_finish` call and a closing reply."""
    return (_tool_call("session_finish", {}, call_id), AIMessage(content=content))


def _tool_call(
    name: str,
    args: dict[str, object],
    call_id: str,
) -> AIMessage:
    """Build one scripted AI tool-call message."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )
