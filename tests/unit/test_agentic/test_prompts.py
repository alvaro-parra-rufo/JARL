"""Tests for agentic system prompts."""

from __future__ import annotations

from jarl.agentic.prompts import (
    CONTINUATION_SYSTEM_PROMPT,
    DEFAULT_NONINTERACTIVE_OBJECTIVE,
    OPERATE_PHASE_RULES,
    OPERATE_SYSTEM_PROMPT,
    SETUP_PHASE_RULES,
    SETUP_SYSTEM_PROMPT,
    build_phase_system_prompt,
)


class TestAgenticSystemPrompts:
    def test_setup_prompt_covers_initialization_rules(self) -> None:
        assert "setup" in SETUP_SYSTEM_PROMPT.lower()
        assert "graph_create_root" in SETUP_SYSTEM_PROMPT
        assert "session_status" in SETUP_SYSTEM_PROMPT

    def test_operate_prompt_covers_training_and_graph_rules(self) -> None:
        assert "operate" in OPERATE_SYSTEM_PROMPT.lower()
        assert "train_run" in OPERATE_SYSTEM_PROMPT
        assert "config_overrides" in OPERATE_SYSTEM_PROMPT
        assert "JSON schema attached to the description of the tool call" in OPERATE_SYSTEM_PROMPT
        assert "algorithm.learning_rate" in OPERATE_SYSTEM_PROMPT
        assert "algorithm.gamma" in OPERATE_SYSTEM_PROMPT
        assert "config_highlights" in OPERATE_SYSTEM_PROMPT
        assert "do not use `graph_create_root`" in OPERATE_SYSTEM_PROMPT.lower()
        assert "graph_reward" in OPERATE_SYSTEM_PROMPT
        assert "graph_set_reward" in OPERATE_SYSTEM_PROMPT
        assert "graph_set_reward" not in SETUP_SYSTEM_PROMPT
        assert "26" not in OPERATE_SYSTEM_PROMPT
        assert "floor_cell" not in OPERATE_SYSTEM_PROMPT.lower()
        assert "cell_entry" not in OPERATE_SYSTEM_PROMPT

    def test_shared_rules_present_in_both_prompts(self) -> None:
        for prompt in (SETUP_SYSTEM_PROMPT, OPERATE_SYSTEM_PROMPT):
            assert "only the tools bound" in prompt.lower()
            assert "do not invent" in prompt.lower()
            assert "ppo" not in prompt.lower()
            assert "current objective" not in prompt.lower()


class TestBuildPhaseSystemPrompt:
    def test_without_objective_matches_phase_constants(self) -> None:
        assert build_phase_system_prompt(SETUP_PHASE_RULES) == SETUP_SYSTEM_PROMPT
        assert build_phase_system_prompt(OPERATE_PHASE_RULES) == OPERATE_SYSTEM_PROMPT

    def test_blank_objective_is_ignored(self) -> None:
        assert build_phase_system_prompt(SETUP_PHASE_RULES, objective="  ") == SETUP_SYSTEM_PROMPT

    def test_objective_appends_non_interactive_goal_without_validation_hints(self) -> None:
        prompt = build_phase_system_prompt(
            SETUP_PHASE_RULES,
            objective=DEFAULT_NONINTERACTIVE_OBJECTIVE,
        )
        objective_block = prompt.removeprefix(SETUP_SYSTEM_PROMPT).lower()

        assert prompt.startswith(SETUP_SYSTEM_PROMPT)
        assert "Current objective:" in prompt
        assert DEFAULT_NONINTERACTIVE_OBJECTIVE in prompt
        assert "session_finish" in prompt
        assert "non-interactive" in objective_block
        assert "do not ask the user" in objective_block
        assert "fail" not in objective_block
        assert "validat" not in objective_block


class TestContinuationSystemPrompt:
    def test_nudge_asks_for_tools_or_finish_without_validation_hints(self) -> None:
        lowered = CONTINUATION_SYSTEM_PROMPT.lower()

        assert "session_finish" in CONTINUATION_SYSTEM_PROMPT
        assert "do not ask the user" in lowered
        assert "fail" not in lowered
        assert "validat" not in lowered
