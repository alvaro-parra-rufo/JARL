"""PPO-only fake script must not pass the EmptyVariant spec case."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from jarl.agentic.cases.builtin.navix_empty_variant_spec import CASE
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel


def test_ppo_only_script_fails_without_set_reward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CASE, "_max_continuations", 0)
    llm = ToolBindingFakeChatModel(
        messages=iter(
            (
                _tool_call("graph_reward", {}, "reward"),
                _tool_call(
                    "graph_extend",
                    {
                        "branch": "main",
                        "label": "ppo_only",
                        "prepare": True,
                        "config_overrides": {"algorithm.learning_rate": 0.001},
                    },
                    "extend_ppo",
                ),
                _tool_call("session_finish", {}, "session_finish"),
                AIMessage(content="Retrained with new PPO hyperparameters."),
            )
        )
    )

    report = CASE.run(tmp_path / "ppo_only", llm=llm)

    assert not report.passed
    assert any("graph_set_reward" in failure for failure in report.failures)


def _tool_call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
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
