"""Contract tests for remaining tool descriptions (skill § Descripciones LLM)."""

from __future__ import annotations

import json

import pytest

from jarl.agentic.prompts import OPERATE_SYSTEM_PROMPT, SETUP_SYSTEM_PROMPT
from jarl.agentic.tools.graph.checkout import TOOL_SPEC as CHECKOUT_SPEC
from jarl.agentic.tools.graph.create_root import TOOL_SPEC as CREATE_ROOT_SPEC
from jarl.agentic.tools.graph.diff import TOOL_SPEC as DIFF_SPEC
from jarl.agentic.tools.graph.reward import TOOL_SPEC as REWARD_SPEC
from jarl.agentic.tools.graph.set_reward import TOOL_SPEC as SET_REWARD_SPEC
from jarl.agentic.tools.graph.set_reward import ToolRequest as SetRewardToolRequest
from jarl.agentic.tools.graph.subtree import TOOL_SPEC as SUBTREE_SPEC
from jarl.agentic.tools.graph.summary import TOOL_SPEC as SUMMARY_SPEC
from jarl.agentic.tools.session.status import TOOL_SPEC as SESSION_SPEC
from jarl.agentic.tools.specs import ToolSpec
from jarl.agentic.tools.subagent.metrics_analysis import TOOL_SPEC as METRICS_SPEC
from jarl.agentic.tools.train.recovery_status import TOOL_SPEC as RECOVERY_SPEC
from jarl.envs.navix.reward_config import RewardWeightsConfig


@pytest.mark.parametrize(
    ("spec", "read_markers"),
    [
        pytest.param(SUMMARY_SPEC, ("read", "does not mutate"), id="graph_summary"),
        pytest.param(REWARD_SPEC, ("read", "does not mutate"), id="graph_reward"),
        pytest.param(SUBTREE_SPEC, ("read", "does not mutate"), id="graph_subtree"),
        pytest.param(DIFF_SPEC, ("read", "does not mutate"), id="graph_diff"),
        pytest.param(SESSION_SPEC, ("read", "does not mutate"), id="session_status"),
        pytest.param(METRICS_SPEC, ("read-only", "does not mutate"), id="subagent_metrics"),
        pytest.param(RECOVERY_SPEC, ("does not mutate", "does not resume"), id="train_recovery"),
    ],
)
def test_read_tools_describe_read_only_effect(
    spec: ToolSpec,
    read_markers: tuple[str, str],
) -> None:
    description = spec.description.lower()
    assert read_markers[0] in description
    assert read_markers[1] in description
    assert "use `" not in description


def test_create_root_describes_setup_only() -> None:
    description = CREATE_ROOT_SPEC.description.lower()

    assert "empty experiment" in description
    assert "does not run training" in description
    assert "not for experiments" in description
    assert "train_run" not in description


def test_checkout_describes_pointer_without_graph_change() -> None:
    description = CHECKOUT_SPEC.description.lower()

    assert "current node" in description
    assert "does not fork" in description
    assert "does not" in description and "train" in description


def test_set_reward_describes_in_place_patch() -> None:
    description = SET_REWARD_SPEC.description.lower()

    assert "reward" in description
    assert "does not fork" in description
    assert "does not" in description and "train" in description
    assert "use `" not in description


_LEAKED_SURFACE = (
    "dopamina",
    "bonus",
    "trap",
    "cell_entry",
    "floor_cell",
    "compatible_env_ids",
    "scenario_reward",
    "25.6",
    "16.6",
    "15.93",
    "r_farm",
    "decenas",
    "g=4",
)


def test_reward_tools_and_prompts_hide_overlay_and_tabular_thresholds() -> None:
    schema = json.dumps(SetRewardToolRequest.model_json_schema()).lower()
    surfaces = " ".join(
        [
            REWARD_SPEC.description,
            SET_REWARD_SPEC.description,
            schema,
            OPERATE_SYSTEM_PROMPT,
            SETUP_SYSTEM_PROMPT,
        ]
    ).lower()

    assert set(SetRewardToolRequest.model_fields) - {"node_id"} == set(RewardWeightsConfig.model_fields)
    assert "maximum" not in schema
    assert schema.count('"minimum": 0') >= 20
    assert "cell_entry" not in schema
    assert "graph_reward" in OPERATE_SYSTEM_PROMPT
    assert "graph_set_reward" in OPERATE_SYSTEM_PROMPT
    assert "graph_set_reward" not in SETUP_SYSTEM_PROMPT
    for token in _LEAKED_SURFACE:
        assert token not in surfaces
