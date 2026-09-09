"""Agentic case: solve the prepared EmptyVariant map."""

from __future__ import annotations

from jarl.agentic.cases.base_agentic_case import (
    DEFAULT_MAX_CONTINUATIONS,
    AgenticCaseRun,
    AgenticTurn,
    BaseAgenticCase,
)
from jarl.agentic.cases.builtin._graph_checkpoint_rollout_analysis import RECORD_VIDEO_LAUNCH_OPTION
from jarl.agentic.cases.builtin._navix_empty_variant_eval import (
    checkpoint_eval_metrics,
    hooked_baseline_metrics,
    latest_saved_checkpoint,
    original_reward_payload,
    overlay_entry_scale,
    reward_payload,
)
from jarl.agentic.cases.validation import ValidationReport
from jarl.agentic.prompts import CONTINUATION_SYSTEM_PROMPT
from jarl.envs.navix.custom.empty_variant import EMPTY_VARIANT_ENV_ID
from jarl.envs.navix.scenario_rewards import resolve_scenario_reward_spec
from jarl.experiments.cases import CaseSpec
from jarl.experiments.cases.builtin.navix_empty_variant import NavixEmptyVariantCase
from jarl.experiments.cases.metadata import ExperimentCaseMetadata
from jarl.experiments.node import NodeWorkspace

__all__ = ["CASE", "NavixEmptyVariantSpecCase"]

_SET_REWARD_TOOL = "graph_set_reward"
_REWARD_TOOL = "graph_reward"
_OVERLAY_SCALE_TOLERANCE = 1e-5
"""Absolute tolerance for the isolated overlay pulse vs registered scale."""

_OBJECTIVE_NOT_REACHED = (
    "El objetivo no fue alcanzado. No preguntes al usuario ni esperes confirmación. Sigue con las tools disponibles."
)
"""Nudge after `session_finish` when the map is not yet solved."""


class NavixEmptyVariantSpecCase(BaseAgenticCase):
    """Validate solving the prepared EmptyVariant map.

    Cheap CI checks tools, mix change, overlay snapshot, and map id. Set
    ``require_trained_eval`` for slow/LLM runs that must also train and beat a
    hooked baseline on success and re-entries.
    """

    require_trained_eval: bool = False
    """When true, require training and pre/post eval (slow / real LLM)."""

    def __init__(self, *, max_continuations: int = DEFAULT_MAX_CONTINUATIONS) -> None:
        """Initialize the EmptyVariant solve case."""
        super().__init__(
            CaseSpec(
                id="navix_empty_variant_spec",
                title="Solve the EmptyVariant map",
                description="Train a policy that solves a prepared EmptyVariant root.",
                tags=frozenset({"graph", "operate", "reward", "train", "navix"}),
                requirements=frozenset({"llm", "navix"}),
            ),
            NavixEmptyVariantCase(),
            (AgenticTurn("Entrena una política que resuelva este mapa. Cuando esté resuelto, termina."),),
            max_continuations=max_continuations,
            launch_options=(RECORD_VIDEO_LAUNCH_OPTION,),
        )

    def continuation_prompt(
        self,
        run: AgenticCaseRun,
        report: ValidationReport,
        *,
        finished: bool,
    ) -> str:
        """Tell the model the objective was not met after a premature finish."""
        del run, report
        if finished:
            return _OBJECTIVE_NOT_REACHED
        return CONTINUATION_SYSTEM_PROMPT

    def validate(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Validate mix redesign; add trained eval when required or present."""
        self._validate_surface(run, report)
        trained = self._trained_workspaces(run)
        if self.require_trained_eval:
            report.check(
                bool(trained),
                "Expected training after changing the reward mix.",
            )
        if trained:
            self._validate_trained(run, report, trained)

    def _validate_surface(self, run: AgenticCaseRun, report: ValidationReport) -> None:
        """Check overlay snapshot, map id, inspect, and mix mutation."""
        metadata_path = run.graph.layout.case_metadata_path
        report.check(metadata_path.is_file(), "Expected case.json in the experiment root.")
        if not metadata_path.is_file():
            return
        snapshot = ExperimentCaseMetadata.load(metadata_path)
        original = original_reward_payload()
        mix_changed = False
        for workspace in run.graph.all_nodes.values():
            config = run.graph.resolve_config(workspace)
            env_id = config.environment.env_id
            report.check(
                env_id == EMPTY_VARIANT_ENV_ID,
                "Expected the experiment to stay on the original environment.",
                halted=True,
            )
            report.check(
                config.environment.scenario_reward_id == snapshot.scenario_reward_id,
                "Expected the experiment case snapshot identifiers to remain unchanged.",
            )
            report.check(
                config.environment.scenario_reward_version == snapshot.scenario_reward_version,
                "Expected the experiment case snapshot identifiers to remain unchanged.",
            )
            payload = reward_payload(config)
            if payload is not None and payload != original:
                mix_changed = True
        report.check(
            any(event.tool == _REWARD_TOOL and event.kind == "read" for event in run.audit_events),
            "Expected graph_reward to inspect the configurable mix.",
        )
        report.check(
            any(event.tool == _SET_REWARD_TOOL and event.kind == "mutation" for event in run.audit_events),
            "Expected graph_set_reward to change the configurable mix.",
        )
        report.check(mix_changed, "Expected the configurable mix to differ from the prepared root.")

    def _validate_trained(
        self,
        run: AgenticCaseRun,
        report: ValidationReport,
        trained: tuple[NodeWorkspace, ...],
    ) -> None:
        """Compare post-fix eval against a hooked baseline; overlay still pays."""
        original = original_reward_payload()
        post = self._post_workspace(run, trained, original)
        report.check(
            post is not None,
            "Expected a trained node whose configurable mix differs from the prepared root.",
        )
        if post is None:
            return
        post_config = run.graph.resolve_config(post)
        spec = resolve_scenario_reward_spec(
            post_config.environment.env_id,
            post_config.environment.scenario_reward_id,
            post_config.environment.scenario_reward_version,
        )
        report.check(
            spec is not None,
            "Expected the experiment case snapshot to remain configured on the trained node.",
        )
        if spec is not None:
            report.check(
                abs(overlay_entry_scale(post_config) - float(spec.scale)) < _OVERLAY_SCALE_TOLERANCE,
                "Expected cell-entry payment from the case snapshot to remain unchanged.",
            )
        pre_workspace = self._pre_workspace(run, trained, original, post)
        if pre_workspace is None:
            root = run.graph.get_node(run.experiment.node_id("root"))
            pre_metrics = hooked_baseline_metrics(run.graph.resolve_config(root))
        else:
            pre_metrics = checkpoint_eval_metrics(
                pre_workspace,
                run.graph.resolve_config(pre_workspace),
            )
        post_metrics = checkpoint_eval_metrics(post, post_config)
        report.check(
            post_metrics.success_rate > pre_metrics.success_rate,
            "Expected higher eval success than the hooked baseline.",
        )
        report.check(
            post_metrics.occupancy_mean < pre_metrics.occupancy_mean,
            "Expected the trained policy to loop less than the hooked baseline.",
        )

    def _trained_workspaces(self, run: AgenticCaseRun) -> tuple[NodeWorkspace, ...]:
        """Return nodes that have at least one saved checkpoint."""
        return tuple(
            workspace for workspace in run.graph.all_nodes.values() if latest_saved_checkpoint(workspace) is not None
        )

    def _post_workspace(
        self,
        run: AgenticCaseRun,
        trained: tuple[NodeWorkspace, ...],
        original: dict[str, float],
    ) -> NodeWorkspace | None:
        """Return the trained node with a redesigned mix, preferring current."""
        changed = [
            workspace for workspace in trained if reward_payload(run.graph.resolve_config(workspace)) != original
        ]
        current = run.graph.current_node
        if current in changed:
            return current
        return changed[-1] if changed else None

    def _pre_workspace(
        self,
        run: AgenticCaseRun,
        trained: tuple[NodeWorkspace, ...],
        original: dict[str, float],
        post: NodeWorkspace,
    ) -> NodeWorkspace | None:
        """Return a hooked trained ancestor when the LLM trained before the fix."""
        hooked = [
            workspace
            for workspace in trained
            if workspace.id != post.id and reward_payload(run.graph.resolve_config(workspace)) == original
        ]
        return hooked[-1] if hooked else None


CASE = NavixEmptyVariantSpecCase()
"""Built-in EmptyVariant solve case."""
