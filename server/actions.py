"""Action execution mixin for MolForge."""

from __future__ import annotations

from typing import Dict, List, Mapping

from .shared import (
    DEFAULT_TOOL_COSTS,
    compute_objective_score,
    evaluate_constraint_margins,
    evaluate_constraints,
    literature_hints,
)

try:
    from ..models import AssayReading, MolForgeAction, RewardComponent
except ImportError:
    from models import AssayReading, MolForgeAction, RewardComponent


class MolForgeActionMixin:
    """Methods that mutate environment state through actions."""

    def _execute_action(
        self,
        action: MolForgeAction,
        reward_components: List[RewardComponent],
        previous_properties: Mapping[str, float],
        previous_score: float,
    ) -> tuple[float, bool]:
        reward = 0.0
        done = False

        if action.action_type == "edit":
            reward += self._apply_edit(action, reward_components, previous_score)
        elif action.action_type == "run_assay":
            reward += self._run_assay(action, reward_components)
        elif action.action_type == "submit":
            reward, done = self._submit(reward_components)
        elif action.action_type == "restart":
            reward += self._restart(reward_components)
        elif action.action_type == "defer":
            reward -= 0.05
            reward_components.append(
                RewardComponent(
                    name="defer",
                    value=-0.05,
                    explanation="Deferring preserves state but lightly penalizes lost project time.",
                )
            )
            self._last_summary = "The team deferred action to gather its thoughts."

        return reward, done

    def _apply_edit(
        self,
        action: MolForgeAction,
        reward_components: List[RewardComponent],
        previous_score: float,
    ) -> float:
        previous_signature = self._molecule_signature()
        previous_fragment = self._molecule[action.slot]  # type: ignore[index]
        safe_defaults = {
            "warhead": "nitrile",
            "hinge": "pyridine",
            "solvent_tail": "morpholine",
            "back_pocket": "methoxy",
        }

        if action.edit_type == "remove":
            self._molecule[action.slot] = safe_defaults[action.slot]  # type: ignore[index]
        else:
            self._molecule[action.slot] = action.fragment  # type: ignore[index]

        new_signature = self._molecule_signature()
        new_properties = self._true_properties()
        new_score = compute_objective_score(new_properties, self._scenario)
        delta = round(new_score - previous_score, 4)
        if self._reward_mode == "dense":
            reward = delta * 2.0
            explanation = (
                f"Updated {action.slot} from {previous_fragment} to {self._molecule[action.slot]}, "
                f"changing the internal objective score by {delta:+.3f}."
            )
        else:
            reward = 0.04 if delta > 0 else (-0.04 if delta < 0 else 0.0)
            explanation = (
                f"Updated {action.slot} from {previous_fragment} to {self._molecule[action.slot]}. "
                "Edit feedback is intentionally coarse; assays and terminal graders provide the main signal."
            )

        reward_components.append(
            RewardComponent(
                name="edit_delta",
                value=round(reward, 4),
                explanation=explanation,
            )
        )

        if new_signature in self._visited_states:
            reward -= 0.35
            reward_components.append(
                RewardComponent(
                    name="loop_penalty",
                    value=-0.35,
                    explanation="This edit revisited a previously explored molecular state.",
                )
            )
        else:
            reward += 0.06
            self._visited_states.add(new_signature)

        reward -= 0.12
        reward_components.append(
            RewardComponent(
                name="turn_cost",
                value=-0.12,
                explanation="Every chemistry edit consumes simulated project time.",
            )
        )
        self._last_summary = (
            f"Lead Chemist edited {action.slot}; molecule changed from "
            f"{previous_signature} to {new_signature}."
        )
        return reward

    def _run_assay(
        self,
        action: MolForgeAction,
        reward_components: List[RewardComponent],
    ) -> float:
        tool_name = action.tool_name or ""
        cost = DEFAULT_TOOL_COSTS[tool_name]
        self._state.remaining_budget -= cost
        self._state.budget_used += cost
        self._state.oracle_call_count += 1

        key = self._assay_context_key(tool_name)
        runs = self._assay_runs.get(key, 0) + 1
        self._assay_runs[key] = runs

        reward = 0.02
        if runs == 1:
            reward += 0.10
            explanation = "First assay on this molecule/tool pair increased observability."
        else:
            reward -= 0.08
            explanation = "Repeated assay spent budget on the same molecule/tool pair."

        readings = self._build_assay_readings(tool_name, runs)
        self._merge_assays(readings)
        if tool_name == "search_literature":
            reward += 0.04
        if self._reward_mode == "curriculum" and runs == 1:
            required_props = {"potency", "toxicity"}
            if "synth_min" in self._scenario.hard_constraints:
                required_props.add("synth")
            covered_props = {
                reading.property_name
                for reading in readings
                if reading.property_name in required_props
            }
            if covered_props:
                bonus = 0.08 * len(covered_props)
                reward += bonus
                reward_components.append(
                    RewardComponent(
                        name="curriculum_evidence_gate",
                        value=round(bonus, 4),
                        explanation=(
                            "Curriculum reward for collecting first-pass evidence "
                            f"for: {', '.join(sorted(covered_props))}."
                        ),
                    )
                )

        reward_components.append(
            RewardComponent(
                name="assay_information_gain",
                value=round(reward, 4),
                explanation=explanation,
            )
        )
        reward_components.append(
            RewardComponent(
                name="budget_spend",
                value=round(-cost / max(self._scenario.oracle_budget, 1), 4),
                explanation=f"Spent {cost} assay budget on {tool_name}.",
            )
        )
        reward -= cost / max(self._scenario.oracle_budget, 1)

        self._oracle_log.append(
            {
                "step": self._state.step_count,
                "tool_name": tool_name,
                "runs": runs,
                "molecule": self._molecule_signature(),
                "context_key": key,
                "cost": cost,
                "results": [reading.model_dump() for reading in readings],
            }
        )
        self._last_summary = (
            f"Assay Planner executed {tool_name}; {len(readings)} structured assay result(s) are now visible."
        )
        return reward

    def _submit(self, reward_components: List[RewardComponent]) -> tuple[float, bool]:
        properties = self._true_properties()
        chemistry = self._chemical_diagnostics()
        final_score = compute_objective_score(properties, self._scenario)
        submission_score = self._grade_submission(properties)
        constraint_results = evaluate_constraints(properties, self._scenario)
        constraint_margins = evaluate_constraint_margins(properties, self._scenario)
        margin_score = sum(constraint_margins.values()) / max(len(constraint_margins), 1)
        violation_penalty = round((1.0 - margin_score) * 2.0, 4)
        hard_constraints_met = all(result[0] for result in constraint_results.values())
        budget_efficiency = self._state.remaining_budget / max(self._scenario.oracle_budget, 1)
        beats_baseline = final_score >= self._scenario.baseline_to_beat
        current_signature = self._molecule_signature()
        evidence_requirements = ["potency", "toxicity"]
        if "synth_min" in self._scenario.hard_constraints:
            evidence_requirements.append("synth")
        missing_evidence = [
            prop for prop in evidence_requirements if self._current_property_estimate(prop, current_signature) is None
        ]
        evidence_met = not missing_evidence
        post_shift_evidence_met = True
        if self._scenario.target_shift_step and self._target_shift_active():
            post_shift_evidence_met = any(
                entry["step"] >= self._scenario.target_shift_step
                and entry["molecule"] == current_signature
                and any(result["property_name"] == "potency" for result in entry["results"])
                for entry in self._oracle_log
            )
        chemistry_gate_met = chemistry.get("passes_filters", True)
        valid_submission = (
            hard_constraints_met
            and beats_baseline
            and evidence_met
            and post_shift_evidence_met
            and chemistry_gate_met
        )

        reward = submission_score * 2.2 if valid_submission else submission_score * 0.20
        if valid_submission:
            reward += 3.5
        elif not hard_constraints_met:
            reward -= violation_penalty
        if not beats_baseline:
            reward -= 0.6
        if not evidence_met:
            reward -= 1.2
        if not post_shift_evidence_met:
            reward -= 0.8
        if not chemistry_gate_met:
            reward -= 0.9

        if valid_submission:
            reward += max(0.0, budget_efficiency) * 0.7
        if self._reward_mode == "curriculum" and evidence_met and post_shift_evidence_met:
            submit_bonus = 0.35
            if hard_constraints_met:
                submit_bonus += 0.15
            reward += submit_bonus

        self._state.submitted = True
        self._report_card = self._build_report_card(submitted=True)
        self._last_summary = (
            f"The team submitted a candidate that "
            f"{'passed' if hard_constraints_met else 'failed'} hard constraints."
        )

        reward_components.extend(
            [
                RewardComponent(
                    name="submission_quality",
                    value=round((submission_score * 2.2 if valid_submission else submission_score * 0.20), 4),
                    explanation=(
                        "Full terminal reward because the submission met scientific, evidence, and chemistry gates."
                        if valid_submission
                        else "Only a small quality trace is awarded because the submit action missed a gate."
                    ),
                ),
                RewardComponent(
                    name="hard_constraints",
                    value=(
                        3.5
                        if valid_submission
                        else (-violation_penalty if not hard_constraints_met else 0.0)
                    ),
                    explanation=(
                        "Large sparse bonus for beating baseline with required current evidence."
                        if valid_submission
                        else "Submission missed constraints, baseline, or evidence requirements; constraint penalty scales with violation severity."
                    ),
                ),
                RewardComponent(
                    name="constraint_margin",
                    value=round(margin_score, 4),
                    explanation=(
                        "Proportional hard-constraint score: worse potency, toxicity, or synthesis violations produce lower values."
                    ),
                ),
                RewardComponent(
                    name="baseline_gate",
                    value=0.0 if beats_baseline else -0.6,
                    explanation=(
                        "Submitted molecule beat the scenario baseline."
                        if beats_baseline
                        else "Submitted molecule did not beat the scenario baseline."
                    ),
                ),
                RewardComponent(
                    name="submission_evidence",
                    value=0.0 if evidence_met else -1.2,
                    explanation=(
                        "Current-molecule potency/toxicity/synthesis evidence was available."
                        if evidence_met
                        else f"Submission lacked current evidence for: {', '.join(missing_evidence)}."
                    ),
                ),
                RewardComponent(
                    name="post_shift_evidence",
                    value=0.0 if post_shift_evidence_met else -0.8,
                    explanation=(
                        "Post-shift potency evidence was available for the submitted molecule."
                        if post_shift_evidence_met
                        else "Hard scenario submission lacked post-shift potency evidence for the current molecule."
                    ),
                ),
                RewardComponent(
                    name="chemical_validity",
                    value=0.0 if chemistry_gate_met else -0.9,
                    explanation=(
                        "Submission passed the medicinal chemistry filter package."
                        if chemistry_gate_met
                        else "Submission failed medicinal chemistry filters despite good surrogate scores."
                    ),
                ),
                RewardComponent(
                    name="budget_efficiency",
                    value=round(max(0.0, budget_efficiency) * 0.7, 4) if valid_submission else 0.0,
                    explanation=(
                        "Unused budget is rewarded to discourage wasteful oracle usage."
                        if valid_submission
                        else "Budget efficiency is not awarded to a gated or premature submission."
                    ),
                ),
            ]
        )
        if self._reward_mode == "curriculum" and evidence_met and post_shift_evidence_met:
            reward_components.append(
                RewardComponent(
                    name="curriculum_evidence_supported_submit",
                    value=round(submit_bonus, 4),
                    explanation=(
                        "Curriculum reward for making a formal submit decision after the required "
                        "current evidence package was available."
                    ),
                )
            )
        return reward, True

    def _restart(self, reward_components: List[RewardComponent]) -> float:
        self._molecule = dict(self._scenario.restart_scaffold)
        self._trap_penalty_active = False
        self._known_assays = []
        self._assay_runs = {}
        self._restart_used = True
        self._visited_states.add(self._molecule_signature())
        self._state.remaining_budget -= 350
        self._state.budget_used += 350
        reward_components.append(
            RewardComponent(
                name="restart_penalty",
                value=-0.4,
                explanation="Restarting discards sunk work but switches to a clean scaffold family.",
            )
        )
        self._last_summary = (
            "The team abandoned the original scaffold series and restarted from a cleaner alternative."
        )
        return -0.4

    def _build_assay_readings(self, tool_name: str, runs: int) -> List[AssayReading]:
        properties = self._true_properties()
        signature = self._molecule_signature()

        if tool_name == "evaluate_properties":
            property_names = ["potency", "novelty", "chemical_quality", "reference_similarity"]
        elif tool_name == "dock_target":
            property_names = ["potency"]
        elif tool_name == "assay_toxicity":
            property_names = ["toxicity"]
        elif tool_name == "estimate_synthesizability":
            property_names = ["synth"]
        elif tool_name == "evaluate_novelty":
            property_names = ["novelty"]
        elif tool_name == "search_literature":
            hint_score = min(0.95, 0.45 + 0.08 * runs)
            return [
                AssayReading(
                    tool_name=tool_name,
                    property_name="literature_signal",
                    estimate=round(hint_score, 4),
                    confidence_low=max(0.0, round(hint_score - 0.08, 4)),
                    confidence_high=min(1.0, round(hint_score + 0.08, 4)),
                    runs=runs,
                    molecule_signature=signature,
                    summary=literature_hints(self._molecule)[0],
                )
            ]
        else:
            property_names = ["potency", "toxicity", "synth"]

        readings = []
        for property_name in property_names:
            true_value = properties[property_name]
            estimate = self._assay_estimate(signature, tool_name, property_name, runs, true_value)
            width = max(0.03, 0.18 / runs)
            readings.append(
                AssayReading(
                    tool_name=tool_name,
                    property_name=property_name,
                    estimate=estimate,
                    confidence_low=max(0.0, round(estimate - width, 4)),
                    confidence_high=min(1.0, round(estimate + width, 4)),
                    runs=runs,
                    molecule_signature=signature,
                    summary=f"{tool_name} estimated {property_name} with run count {runs}.",
                )
            )
        return readings
