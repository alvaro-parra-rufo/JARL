"""Base class for running and deterministically validating agentic cases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from jarl.agentic.audit import AuditIndexEvent, load_audit_index
from jarl.agentic.cases.launch_options import CaseLaunchOption, resolve_tool_overrides
from jarl.agentic.cases.validation import ValidationReport
from jarl.agentic.langgraph.state import AgentState
from jarl.agentic.llm import LLMSettings
from jarl.agentic.progress import append_progress_event
from jarl.agentic.prompts import CONTINUATION_SYSTEM_PROMPT, DEFAULT_NONINTERACTIVE_OBJECTIVE
from jarl.agentic.workflow import AgenticWorkflow
from jarl.experiments.cases import CaseSpec, ExperimentCase, ExperimentCaseContext
from jarl.experiments.graph import ExperimentGraph
from jarl.training.config import RLRunConfig

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "DEFAULT_MAX_CONTINUATIONS",
    "AgenticCaseRun",
    "AgenticTurn",
    "BaseAgenticCase",
]

DEFAULT_MAX_CONTINUATIONS = 3
"""Nudge retries after a user turn when the case is not finished."""


@dataclass(frozen=True, slots=True)
class AgenticTurn:
    """One user prompt sent to `AgenticWorkflow`.

    Args:
        prompt: User message for this workflow turn.
    """

    prompt: str

    def __post_init__(self) -> None:
        """Reject empty prompts."""
        if not self.prompt.strip():
            raise ValueError("Agentic turn prompt must not be empty.")


@dataclass(frozen=True, slots=True)
class AgenticCaseRun:
    """Evidence available to an agentic case validator.

    Args:
        experiment: Materialized initial experiment and logical aliases.
        workflow: Executed workflow with its persisted session.
        graph: Experiment graph reloaded after all turns.
        final_state: LangGraph state returned by the final turn.
        audit_events: Persisted tool audit events for the experiment.
    """

    experiment: ExperimentCaseContext[RLRunConfig]
    workflow: AgenticWorkflow
    graph: ExperimentGraph[RLRunConfig]
    final_state: AgentState
    audit_events: tuple[AuditIndexEvent, ...]

    @property
    def experiment_dir(self) -> Path:
        """Return the materialized experiment directory."""
        return self.experiment.experiment_dir


class BaseAgenticCase(ABC):
    """Reusable agent prompts and deterministic validation over an experiment case."""

    def __init__(
        self,
        spec: CaseSpec,
        experiment_case: ExperimentCase[RLRunConfig],
        turns: tuple[AgenticTurn, ...],
        *,
        audit_reads: bool = True,
        max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
        launch_options: tuple[CaseLaunchOption, ...] = (),
    ) -> None:
        """Initialize an agentic case and its workflow inputs."""
        if not turns:
            raise ValueError("Agentic case must define at least one turn.")
        if max_continuations < 0:
            raise ValueError("max_continuations must be >= 0.")
        self._spec = spec
        self._experiment_case = experiment_case
        self._turns = turns
        self._audit_reads = audit_reads
        self._max_continuations = max_continuations
        self._launch_options = launch_options

    @property
    def spec(self) -> CaseSpec:
        """Return stable case metadata."""
        return self._spec

    @property
    def experiment_case(self) -> ExperimentCase[RLRunConfig]:
        """Return the experiment case materialized before invocation."""
        return self._experiment_case

    @property
    def turns(self) -> tuple[AgenticTurn, ...]:
        """Return the ordered user turns."""
        return self._turns

    @property
    def audit_reads(self) -> bool:
        """Return whether read tools are included in the audit evidence."""
        return self._audit_reads

    @property
    def max_continuations(self) -> int:
        """Return how many generic nudges a user turn may receive."""
        return self._max_continuations

    @property
    def launch_options(self) -> tuple[CaseLaunchOption, ...]:
        """Return optional Runner Lab / CLI controls for this case."""
        return self._launch_options

    def continuation_prompt(
        self,
        run: AgenticCaseRun,
        report: ValidationReport,
        *,
        finished: bool,
    ) -> str:
        """Return the system nudge after a turn that did not close the case.

        The default is `CONTINUATION_SYSTEM_PROMPT`. Overrides must not copy
        ``report.failures``.

        Args:
            run: Evidence from the invoke that just finished.
            report: Fresh validation report for that invoke.
            finished: Whether that invoke called `session_finish`.
        """
        del run, report, finished
        return CONTINUATION_SYSTEM_PROMPT

    def run(
        self,
        destination: str | Path,
        *,
        llm: BaseChatModel | None = None,
        llm_settings: LLMSettings | None = None,
        prompts: Sequence[str] | None = None,
        launch_options: Mapping[str, object] | None = None,
    ) -> ValidationReport:
        """Materialize, invoke, and validate this case.

        After each workflow invoke, validation runs on a fresh report. An
        irreversible failure stops the case. A passing report closes the case
        only when that invoke called `session_finish`. Otherwise a generic
        system nudge is sent, up to `max_continuations` times per user turn.
        The case `objective`, or the default non-interactive objective, is
        passed as the workflow ``objective``. The user turn carries the order.

        Args:
            destination: New or empty directory for the isolated experiment.
            llm: Optional chat model injected into `AgenticWorkflow` (global
                override for the turn; skips catalog assignments including
                subagents).
            llm_settings: Single-profile settings synthesized into a catalog
                when `llm` is omitted.
            prompts: Optional prompt overrides for each turn, in order. When
                omitted, the case's built-in turns are used.
            launch_options: Optional forced tool fields declared by
                ``self.launch_options``.

        Returns:
            Deterministic validation report.

        Raises:
            ValueError: If both an LLM instance and settings are provided, or
                if ``prompts`` length does not match the case turn count.
        """
        if llm is not None and llm_settings is not None:
            raise ValueError("Pass either llm or llm_settings, not both.")
        run_prompts = self._resolve_run_prompts(prompts)
        experiment = self._experiment_case.materialize(destination)
        objective = self._spec.resolved_objective or DEFAULT_NONINTERACTIVE_OBJECTIVE
        append_progress_event(
            experiment.experiment_dir,
            "case_start",
            f"Iniciando caso {self._spec.id}",
            data={"case_id": self._spec.id},
        )
        workflow = AgenticWorkflow.from_experiment(
            experiment.experiment_dir,
            audit_reads=self._audit_reads,
            tool_overrides=resolve_tool_overrides(self._launch_options, launch_options),
        )
        final_state: AgentState | None = None
        report = ValidationReport(case_id=self._spec.id)
        turn_count = len(run_prompts)
        for index, prompt in enumerate(run_prompts, start=1):
            append_progress_event(
                experiment.experiment_dir,
                "turn_start",
                f"Turno {index}/{turn_count}",
                data={"index": index, "prompt": prompt},
            )
            incoming: HumanMessage | SystemMessage = HumanMessage(content=prompt)
            nudge = CONTINUATION_SYSTEM_PROMPT
            for continuation in range(self._max_continuations + 1):
                if continuation:
                    append_progress_event(
                        experiment.experiment_dir,
                        "case_continuation",
                        f"Continuación {continuation}/{self._max_continuations}",
                        data={
                            "index": index,
                            "continuation": continuation,
                            "max_continuations": self._max_continuations,
                        },
                    )
                    incoming = SystemMessage(content=nudge)
                previous_count = 0 if final_state is None else final_state.message_count
                final_state = workflow.invoke(
                    {"messages": [incoming]},
                    llm=llm,
                    llm_settings=llm_settings,
                    objective=objective,
                )
                if continuation == 0:
                    append_progress_event(
                        experiment.experiment_dir,
                        "turn_end",
                        f"Turno {index}/{turn_count} completado",
                        data={"index": index, "message_count": final_state.message_count},
                    )
                run = AgenticCaseRun(
                    experiment=experiment,
                    workflow=workflow,
                    graph=workflow.reload_graph(),
                    final_state=final_state,
                    audit_events=load_audit_index(experiment.experiment_dir),
                )
                report = ValidationReport(case_id=self._spec.id)
                self.validate(run, report)
                if report.halted:
                    append_progress_event(
                        experiment.experiment_dir,
                        "case_halt",
                        "Caso abortado: " + "; ".join(report.failures),
                        data={
                            "index": index,
                            "continuation": continuation,
                            "failures": list(report.failures),
                        },
                    )
                    return report
                finished = final_state.tool_called(
                    "session_finish",
                    after=previous_count,
                )
                if report.passed and finished:
                    append_progress_event(
                        experiment.experiment_dir,
                        "case_finish",
                        "Caso cerrado con session_finish",
                        data={"index": index, "continuation": continuation},
                    )
                    return report
                nudge = self.continuation_prompt(run, report, finished=finished)

        if final_state is None:
            raise RuntimeError("Agentic case completed without invoking any turn.")
        return report

    def _resolve_run_prompts(self, prompts: Sequence[str] | None) -> tuple[str, ...]:
        """Return prompts used for one run, validating optional overrides."""
        if prompts is None:
            return tuple(turn.prompt for turn in self._turns)
        if len(prompts) != len(self._turns):
            msg = f"Expected {len(self._turns)} prompt override(s) for case {self._spec.id!r}, got {len(prompts)}."
            raise ValueError(msg)
        resolved: list[str] = []
        for index, prompt in enumerate(prompts, start=1):
            text = prompt.strip()
            if not text:
                msg = f"Prompt override for turn {index} must not be empty."
                raise ValueError(msg)
            resolved.append(text)
        return tuple(resolved)

    @abstractmethod
    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Evaluate deterministic expectations against one completed run."""
