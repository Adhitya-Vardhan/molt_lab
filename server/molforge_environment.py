"""MolForge environment implementation."""

from __future__ import annotations

import os
import random
from dataclasses import replace
from typing import Any, Dict, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .actions import MolForgeActionMixin
from .governance import MolForgeGovernanceMixin
from .shared import (
    FRAGMENT_LIBRARY,
    SCENARIOS,
    SLOT_ORDER,
    compute_objective_score,
    get_scenario,
)
from .shared import MolForgeSharedMixin
from .views import MolForgeViewMixin

try:
    from ..models import GovernanceStatus, MolForgeAction, MolForgeObservation, MolForgeState, RewardComponent
except ImportError:
    from models import GovernanceStatus, MolForgeAction, MolForgeObservation, MolForgeState, RewardComponent


class MolForgeEnvironment(
    MolForgeActionMixin,
    MolForgeGovernanceMixin,
    MolForgeViewMixin,
    MolForgeSharedMixin,
    Environment,
):
    """Deterministic medicinal-chemistry design environment for OpenEnv."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._debug_state_enabled = os.getenv("MOLFORGE_DEBUG_STATE", "").lower() in {"1", "true", "yes"}
        self._training_randomization_enabled = os.getenv("MOLFORGE_TRAINING_RANDOMIZATION", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self._reward_mode = os.getenv("MOLFORGE_REWARD_MODE", "assay_gated").lower()
        self._rng = random.Random(os.getenv("MOLFORGE_RANDOM_SEED", "molforge"))
        self._reset_index = -1
        self._state = MolForgeState(episode_id=str(uuid4()), step_count=0)
        self._scenario = SCENARIOS[0]
        self._molecule: Dict[str, str] = {}
        self._assay_runs: Dict[str, int] = {}
        self._known_assays: List = []
        self._message_log: List = []
        self._history: List[Dict[str, Any]] = []
        self._oracle_log: List[Dict[str, Any]] = []
        self._visited_states: set[str] = set()
        self._last_summary = ""
        self._report_card = ""
        self._reward_total = 0.0
        self._restart_used = False
        self._trap_penalty_active = False
        self._role_metrics = self._empty_role_metrics()
        self._state_path: List[str] = ["[start]"]
        self._last_governance = GovernanceStatus(
            status="ready",
            explanation="Awaiting the first coordinated decision.",
            required_roles=[],
            approvals=[],
            objections=[],
            vetoes=[],
            executable=True,
        )

    def reset(self) -> MolForgeObservation:
        """Start a new scenario in a deterministic rotation."""

        self._reset_index += 1
        self._scenario = self._select_reset_scenario()
        self._molecule = dict(self._scenario.starting_scaffold)
        self._assay_runs = {}
        self._known_assays = []
        self._message_log = []
        self._history = []
        self._oracle_log = []
        self._visited_states = {self._molecule_signature()}
        self._last_summary = "Episode initialized with a fresh multi-agent review board."
        self._report_card = ""
        self._reward_total = 0.0
        self._restart_used = False
        self._trap_penalty_active = self._scenario.trap_penalty
        self._role_metrics = self._empty_role_metrics()
        self._state_path = ["[start]"]
        self._last_governance = GovernanceStatus(
            status="ready",
            explanation="Lead Chemist should propose the first coordinated action.",
            required_roles=list(self._scenario.required_review_roles),
            approvals=[],
            objections=[],
            vetoes=[],
            executable=True,
        )

        self._state = MolForgeState(
            episode_id=str(uuid4()),
            step_count=0,
            scenario_id=self._scenario.scenario_id,
            difficulty=self._scenario.difficulty,
            state_label="[start]",
            state_path=list(self._state_path),
            coordination_mode=self._scenario.coordination_mode,  # type: ignore[arg-type]
            enabled_roles=list(self._scenario.enabled_roles),
            target_name=self._scenario.target_name,
            current_molecule=self._molecule_signature(),
            remaining_budget=self._scenario.oracle_budget,
            budget_used=0,
            max_budget=self._scenario.oracle_budget,
            visited_states=1,
            known_assay_count=0,
            invalid_action_count=0,
            objection_count=0,
            oracle_call_count=0,
            message_count=0,
            decision_count=0,
            submitted=False,
            reward_total=0.0,
            metadata={},
        )
        self._sync_state_metadata()
        return self._build_observation(reward=0.0, done=False, reward_components=[])

    def _select_reset_scenario(self):
        """Select a deterministic judge scenario or a randomized training variant."""

        scenario = get_scenario(self._reset_index)
        if not self._training_randomization_enabled:
            return scenario

        scenario = self._rng.choice(SCENARIOS)
        budget_scale = self._rng.uniform(0.85, 1.15)
        max_steps_delta = self._rng.choice([-1, 0, 0, 1])
        starting_scaffold = dict(scenario.starting_scaffold)
        if self._rng.random() < 0.35:
            slot = self._rng.choice(SLOT_ORDER)
            choices = [
                fragment
                for fragment in FRAGMENT_LIBRARY[slot]
                if fragment != starting_scaffold[slot]
            ]
            starting_scaffold[slot] = self._rng.choice(choices)
        return replace(
            scenario,
            oracle_budget=max(1, int(round(scenario.oracle_budget * budget_scale))),
            max_steps=max(4, scenario.max_steps + max_steps_delta),
            starting_scaffold=starting_scaffold,
        )

    def step(self, action: MolForgeAction) -> MolForgeObservation:  # type: ignore[override]
        """Execute one coordinated environment action."""

        reward_components: List[RewardComponent] = []
        done = False
        error_code = ""
        self._state.step_count += 1
        self._state.decision_count += 1

        previous_properties = self._true_properties()
        previous_score = compute_objective_score(previous_properties, self._scenario)

        validation_error = self._validate_action(action)
        if validation_error:
            error_code, message = validation_error
            self._state.invalid_action_count += 1
            self._last_governance = GovernanceStatus(
                status="needs_revision",
                explanation=message,
                required_roles=list(self._scenario.required_review_roles),
                approvals=[],
                objections=[],
                vetoes=[],
                executable=False,
            )
            reward_components.append(
                RewardComponent(
                    name="invalid_action",
                    value=-1.0,
                    explanation=message,
                )
            )
            reward = -1.0
            self._last_summary = message
            self._append_state_label("[invalid]")
        else:
            governance, governance_components, policy_veto = self._assess_governance(
                action, previous_properties
            )
            self._last_governance = governance
            reward_components.extend(governance_components)
            reward = sum(component.value for component in governance_components)

            if policy_veto:
                self._last_summary = governance.explanation
                self._append_state_label("[policy_veto]")
            else:
                self._last_governance.status = "executed"
                action_reward, done = self._execute_action(
                    action, reward_components, previous_properties, previous_score
                )
                reward += action_reward
                if not done:
                    reward += self._evaluate_reasoning_consistency(
                        action,
                        previous_properties,
                        self._true_properties(),
                        reward_components,
                    )
                if done and self._state.submitted:
                    self._append_state_label("[submitted]")
                elif not done:
                    self._append_state_label(f"[decision_{self._state.step_count:02d}]")

        if not done and self._state.step_count >= self._scenario.max_steps:
            done = True
            reward_components.append(
                RewardComponent(
                    name="step_limit",
                    value=-0.3,
                    explanation="Episode ended because the maximum decision horizon was reached.",
                )
            )
            reward -= 0.3
            self._report_card = self._build_report_card(submitted=False)
            self._last_summary = "Max-step termination triggered."
            self._append_state_label("[terminated:max_steps]")

        if not done and self._state.remaining_budget <= 0:
            done = True
            reward_components.append(
                RewardComponent(
                    name="budget_exhausted",
                    value=-0.5,
                    explanation="Episode terminated because the oracle budget reached zero.",
                )
            )
            reward -= 0.5
            self._report_card = self._build_report_card(submitted=False)
            self._last_summary = "Budget exhausted before a valid submission."
            self._append_state_label("[terminated:budget]")

        if done and not self._report_card:
            self._report_card = self._build_report_card(submitted=self._state.submitted)

        if done and not self._state.submitted and self._reward_mode == "curriculum":
            reward += self._curriculum_terminal_progress_reward(reward_components)

        reward = round(reward, 4)
        self._reward_total = round(self._reward_total + reward, 4)
        self._state.reward_total = self._reward_total
        self._state.current_molecule = self._molecule_signature()
        self._state.state_label = self._state_path[-1]
        self._state.state_path = list(self._state_path)
        self._state.visited_states = len(self._visited_states)
        self._state.known_assay_count = len(self._known_assays)
        self._state.last_error_code = error_code

        self._history.append(
            {
                "step": self._state.step_count,
                "action": action.model_dump(exclude_none=True),
                "reward": reward,
                "done": done,
                "molecule": self._molecule_signature(),
                "state_label": self._state.state_label,
                "summary": self._last_summary,
                "governance": self._last_governance.model_dump(),
            }
        )
        if done:
            self._report_card = self._build_report_card(submitted=self._state.submitted)
        self._sync_state_metadata()

        return self._build_observation(
            reward=reward,
            done=done,
            reward_components=reward_components,
        )

    def _curriculum_terminal_progress_reward(self, reward_components: List[RewardComponent]) -> float:
        """Give bounded partial credit for near-miss episodes during RL warmup.

        This intentionally does not change the public submission grader. It only
        makes the training reward less sparse when a model builds evidence or a
        chemically plausible candidate but fails to formally submit.
        """

        grader_scores = self._grade_all()
        progress = (
            0.25 * grader_scores["candidate_score"]
            + 0.25 * grader_scores["constraint_margin_score"]
            + 0.25 * grader_scores["evidence_score"]
            + 0.15 * grader_scores["coordination_score"]
            + 0.10 * grader_scores["budget_score"]
        )
        progress = min(0.75, max(0.0, progress))
        reward_components.append(
            RewardComponent(
                name="curriculum_terminal_progress",
                value=round(progress, 4),
                explanation=(
                    "Bounded warmup reward for non-submitted episodes based on candidate quality, "
                    "constraint margin, evidence coverage, coordination, and budget discipline. "
                    "Official submission_score remains 0.0 without a submit action."
                ),
            )
        )
        missed_nomination_penalty = 0.0
        if (
            grader_scores["evidence_score"] >= 0.99
            and grader_scores["constraint_margin_score"] >= 0.9
            and grader_scores["candidate_score"] >= self._scenario.baseline_to_beat
        ):
            missed_nomination_penalty = -0.25
            reward_components.append(
                RewardComponent(
                    name="curriculum_missed_nomination",
                    value=missed_nomination_penalty,
                    explanation=(
                        "The candidate had a strong evidence package near the decision deadline, "
                        "but the team failed to make a formal submit decision."
                    ),
                )
            )
        return progress + missed_nomination_penalty

    @property
    def state(self) -> MolForgeState:
        """Return the current environment state."""

        return self._state
