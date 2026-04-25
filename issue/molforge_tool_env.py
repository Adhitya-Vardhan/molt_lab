# -*- coding: utf-8 -*-
"""MolForge tool environment for TRL's environment_factory GRPO training.

This module wraps the MolForge environment into the interface expected by
``GRPOTrainer(environment_factory=MolForgeToolEnv)``.  The trainer:

1. Creates a ``MolForgeToolEnv()`` instance for each rollout episode.
2. Calls ``reset()`` to start a new scenario (returns an observation string).
3. Auto-discovers public tool methods (``edit_molecule``, ``run_assay``, etc.)
   and exposes them as function-calling tools for the model.
4. Generates model completions, parses tool calls, invokes the matching
   method, and loops until the model stops calling tools or
   ``max_completion_length`` is reached.
5. Reads reward properties from each environment instance after the episode.

Follows the exact same pattern as the official TRL examples:
- https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_grpo.ipynb
- https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Ensure project root is importable ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Bootstrap openenv shim if needed ──────────────────────────────────────
try:
    import openenv  # noqa: F401
except ImportError:
    import openenv_shim  # noqa: F401

from inference_common import (
    MolForgeAction,
    MolForgeObservation,
    attach_reasoning_fields,
    attach_team_messages,
)
from server.molforge_environment import MolForgeEnvironment


# ═══════════════════════════════════════════════════════════════════════════
#  Observation formatter — converts MolForgeObservation → readable text
# ═══════════════════════════════════════════════════════════════════════════

def _format_observation(obs: MolForgeObservation) -> str:
    """Build a concise text summary from a MolForgeObservation."""
    parts = [
        f"Scenario: {obs.scenario_id} ({obs.difficulty})",
        f"Target: {obs.target_name}",
        f"Task: {obs.task_brief}",
        f"Step: {obs.step_index}/{obs.max_steps}",
        f"Budget: {obs.remaining_budget}/{obs.max_budget}",
        f"Molecule: {obs.current_molecule}",
    ]

    # Molecule slots
    if obs.molecule_slots:
        slots_text = ", ".join(f"{s.slot}={s.fragment}" for s in obs.molecule_slots)
        parts.append(f"Slots: {slots_text}")

    # Editable slots
    if obs.editable_slots:
        parts.append(f"Editable: {', '.join(obs.editable_slots)}")

    # Visible metrics
    if obs.visible_metrics:
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in obs.visible_metrics.items())
        parts.append(f"Metrics: {metrics}")

    # Constraint status
    if obs.constraint_status:
        constraints = []
        for c in obs.constraint_status:
            status = "✓" if c.satisfied else ("✗" if c.satisfied is False else "?")
            constraints.append(f"{c.name}({status}, evidence={c.evidence_status})")
        parts.append(f"Constraints: {', '.join(constraints)}")

    # Known assays (last 6)
    if obs.known_assays:
        for reading in obs.known_assays[-6:]:
            parts.append(
                f"  Assay: {reading.tool_name}/{reading.property_name} "
                f"= {reading.estimate:.3f} [{reading.confidence_low:.3f}-{reading.confidence_high:.3f}] "
                f"on {reading.molecule_signature}"
            )

    # Governance status
    gov = obs.governance
    parts.append(f"Governance: {gov.status} — {gov.explanation}")

    # Last transition summary
    if obs.last_transition_summary:
        parts.append(f"Last: {obs.last_transition_summary}")

    # Reward breakdown (if any)
    if obs.reward_breakdown:
        breakdown = ", ".join(f"{rc.name}={rc.value:+.2f}" for rc in obs.reward_breakdown)
        parts.append(f"Reward breakdown: {breakdown}")

    return "\n".join(parts)


def _format_step_result(obs: MolForgeObservation) -> str:
    """Format the result of a step for tool response."""
    parts = []

    # Status line
    if obs.done:
        parts.append("=== EPISODE COMPLETE ===")
    parts.append(f"Step {obs.step_index}/{obs.max_steps} | Budget: {obs.remaining_budget}/{obs.max_budget}")
    parts.append(f"Molecule: {obs.current_molecule}")

    # Governance feedback
    gov = obs.governance
    parts.append(f"Governance: {gov.status}")
    if gov.explanation:
        parts.append(f"  → {gov.explanation}")

    # Last transition
    if obs.last_transition_summary:
        parts.append(f"Summary: {obs.last_transition_summary}")

    # Reward this step
    parts.append(f"Reward: {obs.reward:+.3f}")

    # Reward breakdown
    if obs.reward_breakdown:
        for rc in obs.reward_breakdown:
            parts.append(f"  {rc.name}: {rc.value:+.3f} — {rc.explanation}")

    # Updated constraints
    if obs.constraint_status:
        for c in obs.constraint_status:
            status = "✓" if c.satisfied else ("✗" if c.satisfied is False else "?")
            parts.append(f"  Constraint {c.name}: {status} (evidence={c.evidence_status})")

    # New assay results (last 3)
    if obs.known_assays:
        for reading in obs.known_assays[-3:]:
            parts.append(
                f"  Assay: {reading.tool_name}/{reading.property_name} "
                f"= {reading.estimate:.3f} [{reading.confidence_low:.3f}-{reading.confidence_high:.3f}]"
            )

    if obs.report_card:
        parts.append(f"\nReport Card:\n{obs.report_card}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  MolForgeToolEnv — the environment_factory class
# ═══════════════════════════════════════════════════════════════════════════

class MolForgeToolEnv:
    """OpenEnv-compatible tool environment for MolForge GRPO training.

    Designed for use with ``GRPOTrainer(environment_factory=MolForgeToolEnv)``.
    Each public method (except ``reset``) is auto-discovered as a tool.
    """

    def __init__(self):
        self.env = MolForgeEnvironment()
        self._obs: MolForgeObservation | None = None
        # Main reward signal
        self.reward = 0.0
        self.done = False
        # Per-signal accumulators
        self._step_rewards: list[float] = []
        self._valid_action_count = 0
        self._invalid_action_count = 0
        self._total_steps = 0

    def reset(self, **kwargs) -> str | None:
        """Start a new MolForge drug design episode.

        Returns the initial observation describing the scenario, target,
        current molecule, and available budget.
        """
        self._obs = self.env.reset()
        self.reward = 0.0
        self.done = self._obs.done
        self._step_rewards = []
        self._valid_action_count = 0
        self._invalid_action_count = 0
        self._total_steps = 0
        return _format_observation(self._obs)

    # ── Tool methods (auto-discovered by GRPOTrainer) ──────────────────

    def edit_molecule(
        self,
        slot: str,
        fragment: str,
        edit_type: str,
        rationale: str,
    ) -> str:
        """Edit the molecule by modifying a structural slot with a fragment.

        Args:
            slot: The molecular slot to edit. Must be one of: warhead, hinge, solvent_tail, back_pocket.
            fragment: The fragment identifier to place in the slot.
            edit_type: Type of structural edit. Must be one of: add_fragment, substitute, remove, undo_last_edit.
            rationale: Scientific reasoning for why this edit should improve the molecule.

        Returns:
            Updated observation with reward feedback and molecule state.
        """
        if self.done:
            raise ValueError("Episode is over. No more actions allowed.")

        action = MolForgeAction(
            action_type="edit",
            acting_role="lead_chemist",
            edit_type=edit_type,
            slot=slot,
            fragment=fragment,
            rationale=rationale,
        )
        return self._step_with_action(action)

    def run_assay(self, tool_name: str, rationale: str) -> str:
        """Run a computational assay or oracle tool on the current molecule.

        Args:
            tool_name: The assay to run. Must be one of: evaluate_properties, dock_target, assay_toxicity, estimate_synthesizability, evaluate_novelty, search_literature, run_md_simulation.
            rationale: Why this assay is needed now given the current evidence gaps.

        Returns:
            Assay results and updated observation with new evidence.
        """
        if self.done:
            raise ValueError("Episode is over. No more actions allowed.")

        action = MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name=tool_name,
            rationale=rationale,
        )
        return self._step_with_action(action)

    def submit_molecule(self, rationale: str) -> str:
        """Submit the current molecule as the final drug candidate.

        Args:
            rationale: Why the current molecule meets the scenario objectives and is ready for submission based on available evidence.

        Returns:
            Submission result with final scores and report card.
        """
        if self.done:
            raise ValueError("Episode is over. No more actions allowed.")

        action = MolForgeAction(
            action_type="submit",
            acting_role="lead_chemist",
            rationale=rationale,
        )
        return self._step_with_action(action)

    def restart_episode(self, rationale: str) -> str:
        """Restart the episode with a fresh scaffold.

        Args:
            rationale: Why the current molecular series is not viable and a restart is the best strategy.

        Returns:
            New observation after restart with fresh starting scaffold.
        """
        if self.done:
            raise ValueError("Episode is over. No more actions allowed.")

        action = MolForgeAction(
            action_type="restart",
            acting_role="lead_chemist",
            rationale=rationale,
        )
        return self._step_with_action(action)

    def defer_action(self, rationale: str) -> str:
        """Defer this turn without making a structural change.

        Args:
            rationale: Why no action is the best choice this turn given the current state.

        Returns:
            Updated observation after deferring.
        """
        if self.done:
            raise ValueError("Episode is over. No more actions allowed.")

        action = MolForgeAction(
            action_type="defer",
            acting_role="lead_chemist",
            rationale=rationale,
        )
        return self._step_with_action(action)

    # ── Internal step logic ────────────────────────────────────────────

    def _step_with_action(self, action: MolForgeAction) -> str:
        """Execute an action against the environment and update tracking."""
        assert self._obs is not None, "Must call reset() before stepping."

        # Attach reasoning fields and team messages (governance simulation)
        action = attach_reasoning_fields(self._obs, action)
        action = attach_team_messages(self._obs, action)

        # Step the environment
        self._obs = self.env.step(action)
        step_reward = float(self._obs.reward)

        # Update tracking
        self._step_rewards.append(step_reward)
        self._total_steps += 1
        self.done = self._obs.done

        # Track valid/invalid actions
        if self._obs.governance.status in ("executed",):
            self._valid_action_count += 1
        elif self._obs.governance.status in ("needs_revision", "policy_veto"):
            self._invalid_action_count += 1

        # Update cumulative reward
        self.reward = sum(self._step_rewards)

        return _format_step_result(self._obs)

    # ── Reward properties (read by reward functions) ───────────────────

    @property
    def env_reward(self) -> float:
        """Total cumulative environment reward over the episode."""
        return self.reward

    @property
    def valid_action_reward(self) -> float:
        """Fraction of actions that were valid (not vetoed/rejected)."""
        total = self._valid_action_count + self._invalid_action_count
        if total == 0:
            return 0.0
        return self._valid_action_count / total

    @property
    def progress_reward(self) -> float:
        """Progress based on grader scores if available."""
        if self._obs is None:
            return 0.0
        scores = self._obs.metadata.get("terminal_grader_scores", {})
        if not scores:
            return 0.0
        return float(scores.get("progress_score", scores.get("final_score", 0.0)))

    @property
    def submission_reward(self) -> float:
        """Binary reward: 1.0 if the model successfully submitted, else 0.0."""
        if self._obs is None:
            return 0.0
        scores = self._obs.metadata.get("terminal_grader_scores", {})
        return 1.0 if float(scores.get("submission_score", 0.0)) > 0.0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Reward functions — passed to GRPOTrainer(reward_funcs=[...])
# ═══════════════════════════════════════════════════════════════════════════

def reward_environment(environments, **kwargs) -> list[float]:
    """Main environment reward from MolForge (sum of all step rewards)."""
    return [env.env_reward for env in environments]


def reward_valid_actions(environments, **kwargs) -> list[float]:
    """Fraction of valid actions (not vetoed or rejected by governance)."""
    return [env.valid_action_reward for env in environments]


def reward_progress(environments, **kwargs) -> list[float]:
    """Progress toward solving the scenario (grader-based)."""
    return [env.progress_reward for env in environments]


def reward_submission(environments, **kwargs) -> list[float]:
    """Binary reward for successfully submitting a candidate."""
    return [env.submission_reward for env in environments]
