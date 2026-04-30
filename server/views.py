"""Observation building and scoring mixin for MolForge."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping

from .shared import (
    DEFAULT_TOOL_COSTS,
    EDITABLE_SLOTS,
    ROLE_MESSAGE_TYPES,
    ROLE_PERMISSIONS,
    SCENARIOS,
    SLOT_ORDER,
    compute_objective_score,
    enumerate_candidate_edits,
    evaluate_constraint_margins,
    evaluate_constraints,
    literature_hints,
    molecule_to_smiles,
    oracle_backend_status,
)

try:
    from ..models import ConstraintCheck, MolForgeObservation, MoleculeSlot, RoleObservation
except ImportError:
    from models import ConstraintCheck, MolForgeObservation, MoleculeSlot, RoleObservation


class MolForgeViewMixin:
    """Observation, report-card, and grader methods."""

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        reward_components: List,
    ) -> MolForgeObservation:
        current_signature = self._molecule_signature()
        current_assays = [
            reading for reading in self._known_assays if reading.molecule_signature == current_signature
        ]
        chemistry = self._chemical_diagnostics()
        visible_metrics = {
            "budget_fraction_remaining": round(
                self._state.remaining_budget / max(self._scenario.oracle_budget, 1), 4
            ),
            "current_molecule_assay_count": float(len(current_assays)),
        }
        for property_name in ["potency", "toxicity", "synth", "novelty"]:
            estimate = self._current_property_estimate(property_name, current_signature)
            if estimate is not None:
                visible_metrics[property_name] = estimate
        if chemistry.get("available"):
            visible_metrics["chemical_quality"] = float(chemistry.get("chemical_quality", 0.5))
            visible_metrics["reference_similarity"] = float(chemistry.get("reference_similarity", 0.5))

        constraint_status = self._build_visible_constraints(current_signature)
        metadata: Dict[str, Any] = {
            "task_index": self._reset_index % len(SCENARIOS),
            "scenario_variant": deepcopy(self._scenario_context),
            "oracle_budget_costs": deepcopy(DEFAULT_TOOL_COSTS),
            "history_length": len(self._history),
            "trace_tail": [entry["summary"] for entry in self._history[-3:]],
            "current_smiles": molecule_to_smiles(self._molecule),
            "chemical_diagnostics": chemistry,
            "oracle_backend": oracle_backend_status(),
            "candidate_edits": [
                {"slot": slot, "fragment": fragment}
                for slot, fragment in list(enumerate_candidate_edits(self._molecule))[:8]
            ],
            "literature_hints": literature_hints(self._molecule),
            "target_shift_active": self._target_shift_active(),
            "public_role_metrics": {
                role: {
                    "messages_sent": metrics["messages_sent"],
                    "correct_messages": metrics["correct_messages"],
                }
                for role, metrics in self._role_metrics.items()
            },
        }
        if done:
            metadata["terminal_grader_scores"] = self._grade_all()

        return MolForgeObservation(
            scenario_id=self._scenario.scenario_id,
            difficulty=self._scenario.difficulty,
            state_label=self._state.state_label,
            state_path=list(self._state_path),
            coordination_mode=self._scenario.coordination_mode,  # type: ignore[arg-type]
            enabled_roles=list(self._scenario.enabled_roles),
            task_brief=self._scenario.task_brief,
            target_name=self._scenario.target_name,
            current_molecule=current_signature,
            molecule_slots=[
                MoleculeSlot(slot=slot, fragment=self._molecule[slot], editable=True)
                for slot in SLOT_ORDER
            ],
            editable_slots=list(EDITABLE_SLOTS),
            step_index=self._state.step_count,
            max_steps=self._scenario.max_steps,
            remaining_budget=self._state.remaining_budget,
            budget_used=self._state.budget_used,
            max_budget=self._scenario.oracle_budget,
            known_assays=deepcopy(self._known_assays),
            role_observations=self._build_role_observations(current_signature),
            message_log=[message.model_dump() for message in self._message_log[-8:]],
            governance=deepcopy(self._last_governance),
            last_transition_summary=self._last_summary,
            visible_metrics=visible_metrics,
            constraint_status=constraint_status,
            reward_breakdown=reward_components,
            allowed_actions=[
                "Lead Chemist: edit, submit, restart, defer",
                "Assay Planner: run_assay",
                "Messages: proposal, approval, objection, risk_flag, assay_request, rejection",
            ],
            report_card=self._report_card,
            metadata=metadata,
            done=done,
            reward=reward,
        )

    def _build_visible_constraints(self, molecule_signature: str) -> List[ConstraintCheck]:
        checks: List[ConstraintCheck] = []
        for name, threshold in self._scenario.hard_constraints.items():
            property_name = "toxicity" if name == "toxicity_max" else name.split("_")[0]
            estimate = self._current_property_estimate(property_name, molecule_signature)
            relation = "<=" if name.endswith("_max") else ">="
            if estimate is None:
                checks.append(
                    ConstraintCheck(
                        name=name,
                        target=f"{relation} {threshold:.2f}",
                        satisfied=None,
                        actual=None,
                        evidence_status="unknown",
                    )
                )
                continue
            satisfied = estimate <= threshold if name.endswith("_max") else estimate >= threshold
            checks.append(
                ConstraintCheck(
                    name=name,
                    target=f"{relation} {threshold:.2f}",
                    satisfied=satisfied,
                    actual=round(estimate, 4),
                    evidence_status="known",
                )
            )
        return checks

    def _build_role_observations(self, molecule_signature: str) -> List[RoleObservation]:
        current_assays = [
            reading.model_dump()
            for reading in self._known_assays
            if reading.molecule_signature == molecule_signature
        ]
        chemistry = self._chemical_diagnostics()
        evidence_gaps = [
            prop
            for prop in ["potency", "toxicity", "synth"]
            if self._current_property_estimate(prop, molecule_signature) is None
        ]
        edit_history = [
            entry["action"]
            for entry in self._history
            if entry["action"].get("action_type") == "edit"
        ][-4:]

        return [
            RoleObservation(
                role="lead_chemist",
                local_objective="Propose high-value scaffold edits and decide when the team should submit.",
                permissions=ROLE_PERMISSIONS["lead_chemist"],
                observation={
                    "molecule_slots": deepcopy(self._molecule),
                    "edit_history": edit_history,
                    "visible_assays": current_assays,
                    "chemical_diagnostics": chemistry,
                    "candidate_edits": [
                        {"slot": slot, "fragment": fragment}
                        for slot, fragment in list(enumerate_candidate_edits(self._molecule))[:8]
                    ],
                    "open_questions": evidence_gaps,
                },
            ),
            RoleObservation(
                role="toxicologist",
                local_objective="Protect against safety regressions and unsafe submissions.",
                permissions=ROLE_MESSAGE_TYPES["toxicologist"],
                observation={
                    "toxicity_readouts": [
                        reading
                        for reading in current_assays
                        if reading["property_name"] == "toxicity"
                    ],
                    "hard_threshold": self._scenario.hard_constraints.get("toxicity_max"),
                    "safety_alerts": self._safety_alerts(),
                    "chemical_diagnostics": chemistry,
                    "risk_history": [
                        message.model_dump()
                        for message in self._message_log
                        if message.sender == "toxicologist"
                    ][-4:],
                },
            ),
            RoleObservation(
                role="assay_planner",
                local_objective="Allocate assay budget where the expected information gain is highest.",
                permissions=ROLE_PERMISSIONS["assay_planner"] + ROLE_MESSAGE_TYPES["assay_planner"],
                observation={
                    "budget_ledger": {
                        "remaining_budget": self._state.remaining_budget,
                        "budget_used": self._state.budget_used,
                        "max_budget": self._state.max_budget,
                    },
                    "tool_costs": deepcopy(DEFAULT_TOOL_COSTS),
                    "tool_usage_history": deepcopy(self._assay_runs),
                    "evidence_gaps": evidence_gaps,
                    "estimated_information_value": {
                        tool_name: round(self._estimate_information_gain(tool_name), 4)
                        for tool_name in self._scenario.enabled_tools
                    },
                },
            ),
            RoleObservation(
                role="process_chemist",
                local_objective="Guard tractability and synthetic feasibility before the team commits.",
                permissions=ROLE_MESSAGE_TYPES["process_chemist"],
                observation={
                    "synth_readouts": [
                        reading for reading in current_assays if reading["property_name"] == "synth"
                    ],
                    "route_warnings": self._route_warnings(),
                    "chemical_diagnostics": chemistry,
                    "feasibility_flags": {
                        "heavy_hinge": self._molecule["hinge"] == "quinazoline",
                        "reactive_warhead": self._molecule["warhead"] == "vinyl_sulfonamide",
                        "lipophilic_tail": self._molecule["back_pocket"] == "trifluoromethyl",
                    },
                },
            ),
        ]

    def _grade_all(self) -> Dict[str, float]:
        properties = self._true_properties()
        constraints = evaluate_constraints(properties, self._scenario)
        constraint_margins = evaluate_constraint_margins(properties, self._scenario)
        constraint_margin_score = sum(constraint_margins.values()) / max(len(constraint_margins), 1)
        constraint_fraction = sum(1.0 for passed, _ in constraints.values() if passed) / max(len(constraints), 1)
        submitted = self._state.submitted
        coordination_score = self._coordination_score()
        evidence_score = self._evidence_score()
        chemical_quality_score = self._chemical_quality_score()
        budget_score = self._open_unit_interval(
            self._state.remaining_budget / max(self._scenario.oracle_budget, 1),
        )
        progress_score = self._grade_progress(
            candidate_score=compute_objective_score(properties, self._scenario),
            constraint_margin_score=constraint_margin_score,
            constraint_fraction=constraint_fraction,
            evidence_score=evidence_score,
            chemical_quality_score=chemical_quality_score,
            coordination_score=coordination_score,
            budget_score=budget_score,
        )
        submission_score = self._grade_submission(properties) if submitted else 0.0
        final_score = self._grade_final(
            submission_score=submission_score,
            progress_score=progress_score,
            submitted=submitted,
            constraint_fraction=constraint_fraction,
            evidence_score=evidence_score,
        )
        return {
            "final_score": final_score,
            "potency_score": self._open_unit_interval(properties["potency"]),
            "safety_score": self._open_unit_interval(1.0 - properties["toxicity"]),
            "synth_score": self._open_unit_interval(properties["synth"]),
            "novelty_score": self._open_unit_interval(properties["novelty"]),
            "candidate_score": self._open_unit_interval(compute_objective_score(properties, self._scenario)),
            "constraint_score": self._open_unit_interval(
                sum(1.0 for passed, _ in constraints.values() if passed) / max(len(constraints), 1),
            ),
            "constraint_margin_score": self._open_unit_interval(constraint_margin_score),
            "budget_score": budget_score,
            "submitted_score": 1.0 if submitted else 0.0,
            "submission_score": submission_score,
            "progress_score": progress_score,
            "chemical_quality_score": self._open_unit_interval(chemical_quality_score),
            "coordination_score": self._open_unit_interval(coordination_score),
            "evidence_score": self._open_unit_interval(evidence_score),
            "reference_similarity_score": self._open_unit_interval(properties.get("reference_similarity", 0.5)),
        }

    def _grade_progress(
        self,
        *,
        candidate_score: float,
        constraint_margin_score: float,
        constraint_fraction: float,
        evidence_score: float,
        chemical_quality_score: float,
        coordination_score: float,
        budget_score: float,
    ) -> float:
        """Score scientific progress even when no formal submission happened."""

        progress = (
            0.45 * candidate_score
            + 0.28 * constraint_margin_score
            + 0.12 * evidence_score
            + 0.08 * chemical_quality_score
            + 0.04 * coordination_score
            + 0.03 * budget_score
        )
        repeated_assays = sum(max(0, runs - 1) for runs in self._assay_runs.values())
        policy_vetoes = sum(
            1
            for entry in self._history
            if entry.get("governance", {}).get("status") == "policy_veto"
        )
        progress -= min(0.20, 0.04 * repeated_assays)
        progress -= min(0.20, 0.05 * policy_vetoes)

        if constraint_fraction < 1.0:
            progress = min(progress, 0.25 + 0.25 * constraint_fraction)
        if not self._state.submitted and evidence_score < 0.99:
            progress = min(progress, 0.45)
        if self._scenario.trap_penalty and not self._restart_used:
            progress = min(progress, 0.30)
        if self._state.submitted:
            progress += 0.05
        return self._open_unit_interval(progress)

    def _grade_final(
        self,
        *,
        submission_score: float,
        progress_score: float,
        submitted: bool,
        constraint_fraction: float,
        evidence_score: float,
    ) -> float:
        """Single conservative scalar for RL/evaluation headline reporting."""

        if submitted:
            return self._open_unit_interval(submission_score)

        score = 0.35 * progress_score
        if constraint_fraction < 1.0:
            score = min(score, 0.05 + 0.10 * constraint_fraction)
        if evidence_score < 0.99:
            score = min(score, 0.15)
        if self._scenario.trap_penalty and not self._restart_used:
            score = min(score, 0.08)
        return self._open_unit_interval(score)

    def _coordination_score(self) -> float:
        expected_messages = 0
        for entry in self._history:
            action = entry.get("action", {})
            if action.get("action_type") == "defer":
                continue
            expected_messages += 1 + len(entry.get("governance", {}).get("required_roles", []))
        if expected_messages == 0:
            return self._open_unit_interval(0.0)
        total_correct = sum(metrics["correct_messages"] for metrics in self._role_metrics.values())
        return self._open_unit_interval(min(total_correct, expected_messages) / expected_messages)

    def _chemical_quality_score(self) -> float:
        return self._open_unit_interval(self._true_properties().get("chemical_quality", 0.5))

    def _grade_submission(self, properties: Mapping[str, float]) -> float:
        base = compute_objective_score(properties, self._scenario)
        chemistry = self._chemical_diagnostics()
        chemical_quality = self._chemical_quality_score()
        constraint_margins = evaluate_constraint_margins(properties, self._scenario)
        constraint_margin_score = sum(constraint_margins.values()) / max(len(constraint_margins), 1)
        constraints = evaluate_constraints(properties, self._scenario)
        constraint_fraction = sum(1.0 for passed, _ in constraints.values() if passed) / max(len(constraints), 1)
        submission_score = (
            0.62 * base
            + 0.20 * constraint_margin_score
            + 0.10 * self._evidence_score()
            + 0.05 * chemical_quality
            + 0.03 * self._coordination_score()
        )
        evidence_score = self._evidence_score()
        if evidence_score >= 0.99 and constraint_fraction >= 1.0 and base >= self._scenario.baseline_to_beat:
            budget_efficiency = self._state.remaining_budget / max(self._scenario.oracle_budget, 1)
            submission_score += 0.05 * max(0.0, budget_efficiency)
        if evidence_score < 1.0:
            submission_score = min(submission_score, 0.25 + 0.25 * evidence_score)
        if constraint_fraction < 1.0:
            submission_score = min(submission_score, 0.20 + 0.50 * constraint_margin_score)
        if base < self._scenario.baseline_to_beat:
            submission_score = min(submission_score, 0.45)
        if chemistry.get("available") and not chemistry.get("passes_filters", True):
            submission_score = min(submission_score, 0.15 + 0.55 * chemical_quality)
        return self._open_unit_interval(submission_score)

    def _evidence_score(self) -> float:
        current_signature = self._molecule_signature()
        required = ["potency", "toxicity"]
        if "synth_min" in self._scenario.hard_constraints:
            required.append("synth")
        available = sum(
            1
            for prop in required
            if self._current_property_estimate(prop, current_signature) is not None
        )
        score = available / max(len(required), 1)
        if self._scenario.target_shift_step and self._target_shift_active():
            has_post_shift_potency = any(
                entry["step"] >= self._scenario.target_shift_step
                and entry["molecule"] == current_signature
                and any(result["property_name"] == "potency" for result in entry["results"])
                for entry in self._oracle_log
            )
            score = min(score, 1.0 if has_post_shift_potency else 0.5)
        return score

    def _build_report_card(self, *, submitted: bool) -> str:
        properties = self._true_properties()
        grader_scores = self._grade_all()
        constraints = evaluate_constraints(properties, self._scenario)
        lines = [
            f"Scenario: {self._scenario.scenario_id} ({self._scenario.difficulty})",
            f"Scenario family: {self._scenario.scenario_family or self._scenario.scenario_id} [{self._scenario.variant_kind}]",
            f"Final molecule: {self._molecule_signature()}",
            f"Potency: {properties['potency']:.3f}",
            f"Toxicity: {properties['toxicity']:.3f}",
            f"Synthesizability: {properties['synth']:.3f}",
            f"Novelty: {properties['novelty']:.3f}",
            f"Chemical quality: {grader_scores['chemical_quality_score']:.3f}",
            f"Final score: {grader_scores['final_score']:.3f}",
            f"Candidate scientific score: {grader_scores['candidate_score']:.3f}",
            f"Constraint margin score: {grader_scores['constraint_margin_score']:.3f}",
            f"Submission grader: {grader_scores['submission_score']:.3f}",
            f"Progress score: {grader_scores['progress_score']:.3f}",
            f"Coordination score: {grader_scores['coordination_score']:.3f}",
            f"Evidence score: {grader_scores['evidence_score']:.3f}",
            f"Reference ligand similarity: {grader_scores['reference_similarity_score']:.3f}",
            "Constraints:",
        ]
        for name, (passed, threshold) in constraints.items():
            metric_name = "toxicity" if name == "toxicity_max" else name.split("_")[0]
            lines.append(
                f"- {name}: {'pass' if passed else 'fail'} (actual={properties[metric_name]:.3f}, threshold={threshold:.3f})"
            )
        chemistry = self._chemical_diagnostics()
        if chemistry.get("available"):
            descriptor_bits = ", ".join(
                f"{name}={value}"
                for name, value in chemistry.get("descriptors", {}).items()
                if name in {"mol_wt", "logp", "tpsa", "rotatable_bonds"}
            )
            lines.append(f"Chemistry diagnostics: {descriptor_bits}")
            alerts = chemistry.get("alerts", [])
            lines.append(f"Chemistry alerts: {', '.join(alerts) if alerts else 'none'}")
            failed_filters = chemistry.get("failed_filters", [])
            if failed_filters:
                lines.append(f"Chemistry filter failures: {', '.join(failed_filters)}")
        lines.append(
            f"Messages sent: {self._state.message_count}, objections raised: {self._state.objection_count}, oracle calls: {self._state.oracle_call_count}"
        )
        if self._scenario.target_shift_step and self._target_shift_active():
            lines.append("Target mutation triggered during this episode.")
        if self._restart_used:
            lines.append("Agent used restart_from_new_scaffold to escape the original trap series.")
        if not submitted:
            lines.append("Episode terminated without a formal submit action.")
        return "\n".join(lines)
