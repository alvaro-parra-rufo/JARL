"""Reactive fake LLM that forks from ``latest`` via ``graph_checkpoints``."""

from __future__ import annotations

import json
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from jarl.experiments.io.checkpoints import CHECKPOINT_ALIAS_LATEST
from tests.helpers.fake_chat_model import ToolBindingFakeChatModel

__all__ = ["ForkFromLatestNotBestFake"]


class ForkFromLatestNotBestFake(ToolBindingFakeChatModel):
    """Call ``graph_checkpoints``, then ``graph_fork`` from the ``latest`` alias."""

    phase: int = 0

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        if self.phase == 0:
            self.phase = 1
            message = _tool_call("graph_checkpoints", {}, "checkpoints_overview")
        elif self.phase == 1:
            self.phase = 2
            node_id, checkpoint_step = _latest_from_messages(messages)
            message = _tool_call(
                "graph_fork",
                {
                    "branch": "from_latest",
                    "label": "latest_child",
                    "prepare": True,
                    "from_checkpoint": {
                        "node_id": node_id,
                        "checkpoint_step": checkpoint_step,
                    },
                },
                "fork_latest",
            )
        elif self.phase == 2:
            self.phase = 3
            message = _tool_call("session_finish", {}, "finish")
        else:
            message = AIMessage(content="Forked from the latest checkpoint, not best.")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_call(name: str, args: dict[str, object], call_id: str) -> AIMessage:
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


def _latest_from_messages(messages: list[BaseMessage]) -> tuple[str, int]:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        payload = json.loads(str(message.content))
        if not isinstance(payload, dict):
            continue
        node_id = payload.get("node_id")
        latest = payload.get("latest")
        if not isinstance(node_id, str) or not isinstance(latest, dict):
            continue
        aliases = latest.get("aliases") or []
        step = latest.get("checkpoint_step")
        if CHECKPOINT_ALIAS_LATEST in aliases and isinstance(step, int):
            return node_id, step
        msg = "graph_checkpoints overview did not expose a latest alias."
        raise AssertionError(msg)
    msg = "Expected a graph_checkpoints ToolMessage before forking from latest."
    raise AssertionError(msg)
