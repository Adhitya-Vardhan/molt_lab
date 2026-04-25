"""Governance, validation, and coordination logic for MolForge."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .shared import (
    DEFAULT_TOOL_COSTS,
    EDITABLE_SLOTS,
    FRAGMENT_LIBRARY,
    ROLE_MESSAGE_TYPES,
    ROLE_PERMISSIONS,
)

try:
    from ..models import GovernanceStatus, MolForgeAction, RewardComponent
except ImportError:
    from models import GovernanceStatus, MolForgeAction, RewardComponent


class MolForgeGovernanceMixin:
    """Validation and multi-agent review methods."""

    def _validate_action(self, action: MolForgeAction) -> Optional[tuple[str, str]]:
        if action.action_type not in self._scenario.enabled_actions:
            return "ACTION_DISABLED", f"{action.action_type} is disabled for this scenario."

        if action.acting_role not in self._scenario.enabled_roles:
            return "ROLE_DISABLED", f"{action.acting_role} is not enabled for this scenario."

        allowed_actions = ROLE_PERMISSIONS.get(action.acting_role, [])
        if action.action_type not in allowed_actions:
            return (
                "ROLE_PERMISSION_DENIED",
                f"{action.acting_role} is not permitted to execute {action.action_type}.",
            )

        if len(action.messages) > self._scenario.max_messages_per_turn:
            return (
                "MESSAGE_LIMIT_EXCEEDED",
                f"At most {self._scenario.max_messages_per_turn} messages may be sent per turn.",
            )

        seen_senders = set()
        for message in action.messages:
            if message.sender not in self._scenario.enabled_roles:
                return "MESSAGE_ROLE_INVALID", f"{message.sender} is not enabled in this scenario."
            if message.sender in seen_senders:
                return (
                    "DUPLICATE_ROLE_MESSAGE",
                    f"Each specialist may emit at most one message per turn; duplicate from {message.sender}.",
                )
            seen_senders.add(message.sender)
            if message.message_type not in ROLE_MESSAGE_TYPES.get(message.sender, []):
                return (
                    "MESSAGE_PERMISSION_DENIED",
                    f"{message.sender} cannot emit message type {message.message_type}.",
                )

        if action.action_type == "edit":
            if action.slot is None or action.edit_type is None:
                return "MISSING_EDIT_FIELDS", "Edit actions require both slot and edit_type."
            if action.slot not in EDITABLE_SLOTS:
                return "INVALID_SLOT", f"{action.slot} is not editable in MolForge."
            if action.edit_type in {"add_fragment", "substitute"} and not action.fragment:
                return "MISSING_FRAGMENT", "Edit actions require a fragment for add/substitute."
            if action.fragment:
                if action.fragment not in FRAGMENT_LIBRARY[action.slot]:
                    return "UNKNOWN_FRAGMENT", f"{action.fragment} is not valid for slot {action.slot}."
                if self._molecule[action.slot] == action.fragment:
                    return "NO_STATE_CHANGE", "Edit selected the fragment already present in that slot."

        if action.action_type == "run_assay":
            if action.tool_name is None:
                return "MISSING_TOOL_NAME", "run_assay actions require a tool_name."
            if action.tool_name not in self._scenario.enabled_tools:
                return "TOOL_DISABLED", f"{action.tool_name} is not enabled for this scenario."
            cost = DEFAULT_TOOL_COSTS[action.tool_name]
            if self._state.remaining_budget < cost:
                return "BUDGET_EXCEEDED", f"{action.tool_name} costs {cost}, exceeding remaining budget."

        if action.action_type == "restart":
            if self._restart_used:
                return "RESTART_ALREADY_USED", "restart_from_new_scaffold may be used at most once per episode."
            if self._state.remaining_budget < 350:
                return "BUDGET_EXCEEDED", "Not enough budget remains to restart from a new scaffold."

        return None

    def _assess_governance(
        self,
        action: MolForgeAction,
        previous_properties: Mapping[str, float],
    ) -> tuple[GovernanceStatus, List[RewardComponent], bool]:
        reward_components: List[RewardComponent] = []
        approvals: List[str] = []
        objections: List[str] = []
        vetoes: List[str] = []
        required_roles = (
            []
            if action.action_type == "defer"
            else [role for role in self._scenario.required_review_roles if role != action.acting_role]
        )
        policy_veto = False

        current_signature = self._molecule_signature()
        simulated_properties = self._simulate_action_properties(action)
        sender_map = {message.sender: message for message in action.messages}

        actor_message = sender_map.get(action.acting_role)
        if action.action_type != "defer":
            if actor_message and actor_message.message_type == "proposal":
                self._record_message(actor_message)
                reward_components.append(
                    RewardComponent(
                        name="proposal_logged",
                        value=0.05,
                        explanation=f"{action.acting_role} logged a structured proposal before execution.",
                    )
                )
                self._role_metrics[action.acting_role]["correct_messages"] += 1
            else:
                reward_components.append(
                    RewardComponent(
                        name="missing_proposal",
                        value=-0.06,
                        explanation="The acting specialist did not provide an explicit proposal message.",
                    )
                )

        for role in required_roles:
            expected = self._expected_feedback(role, action, previous_properties, simulated_properties)
            actual = sender_map.get(role)
            if actual is None:
                reward_components.append(
                    RewardComponent(
                        name=f"missing_review_{role}",
                        value=-0.08,
                        explanation=f"{role} did not provide the required review for this turn.",
                    )
                )
                if expected["hard_veto"]:
                    policy_veto = True
                    vetoes.append(role)
                continue

            if role != action.acting_role:
                self._record_message(actual)
            if self._matches_feedback(actual.message_type, expected["type"]):
                reward_components.append(
                    RewardComponent(
                        name=f"coordination_{role}",
                        value=0.12,
                        explanation=expected["reason"],
                    )
                )
                self._role_metrics[role]["correct_messages"] += 1
                if expected["type"] in {"approval", "submission_recommendation"}:
                    approvals.append(role)
                else:
                    objections.append(role)
            elif expected["type"] == "neutral":
                reward_components.append(
                    RewardComponent(
                        name=f"unnecessary_message_{role}",
                        value=-0.02,
                        explanation=f"{role} contributed a message even though no strong intervention was needed.",
                    )
                )
                self._role_metrics[role]["incorrect_messages"] += 1
            else:
                reward_components.append(
                    RewardComponent(
                        name=f"misaligned_review_{role}",
                        value=-0.1,
                        explanation=(
                            f"{role} sent {actual.message_type}, but the hidden environment evaluation "
                            f"expected {expected['type']}."
                        ),
                    )
                )
                self._role_metrics[role]["incorrect_messages"] += 1
                if expected["hard_veto"]:
                    policy_veto = True
                    vetoes.append(role)

            if expected["hard_veto"] and actual and self._matches_feedback(actual.message_type, expected["type"]):
                policy_veto = True
                vetoes.append(role)

        extra_roles = {
            sender
            for sender in sender_map
            if sender not in required_roles and sender != action.acting_role
        }
        for role in sorted(extra_roles):
            self._record_message(sender_map[role])
            reward_components.append(
                RewardComponent(
                    name=f"optional_review_{role}",
                    value=0.02,
                    explanation=f"{role} added optional context for the current decision.",
                )
            )
            self._role_metrics[role]["correct_messages"] += 1

        if policy_veto:
            reward_components.append(
                RewardComponent(
                    name="policy_veto",
                    value=-0.35,
                    explanation="A specialist raised a valid hard veto, so the action was blocked.",
                )
            )

        explanation = (
            "Team review approved the decision."
            if not policy_veto
            else f"Action blocked after hard veto from {', '.join(vetoes)}."
        )
        status = "policy_veto" if policy_veto else "ready"
        return (
            GovernanceStatus(
                status=status,
                explanation=explanation,
                required_roles=required_roles,
                approvals=approvals,
                objections=objections,
                vetoes=vetoes,
                executable=not policy_veto,
            ),
            reward_components,
            policy_veto,
        )

    def _expected_feedback(
        self,
        role: str,
        action: MolForgeAction,
        previous_properties: Mapping[str, float],
        simulated_properties: Mapping[str, float],
    ) -> Dict[str, Any]:
        tox_threshold = self._scenario.hard_constraints.get("toxicity_max", 0.45)
        synth_threshold = self._scenario.hard_constraints.get("synth_min", 0.56)
        current_signature = self._molecule_signature()
        known_toxicity = self._current_property_estimate("toxicity", current_signature)
        known_synth = self._current_property_estimate("synth", current_signature)

        neutral = {"type": "neutral", "hard_veto": False, "reason": f"{role} had no strong intervention."}

        if role == "toxicologist":
            if action.action_type == "run_assay":
                if action.tool_name == "assay_toxicity":
                    return {
                        "type": "approval",
                        "hard_veto": False,
                        "reason": "Toxicologist correctly encouraged a direct safety assay.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Toxicologist correctly allowed an assay that improves decision evidence without increasing chemical risk.",
                }
            if action.action_type == "submit":
                if known_toxicity is not None and known_toxicity > tox_threshold:
                    return {
                        "type": "objection",
                        "hard_veto": simulated_properties["toxicity"] > tox_threshold,
                        "reason": "Toxicologist correctly objected to a submission with visible toxicity evidence above threshold.",
                    }
                if simulated_properties["toxicity"] > tox_threshold:
                    return {
                        "type": "objection",
                        "hard_veto": True,
                        "reason": "Toxicologist correctly blocked an unsafe submission.",
                    }
                if known_toxicity is None:
                    return {
                        "type": "assay_request",
                        "hard_veto": True,
                        "reason": "Toxicologist correctly demanded explicit toxicity evidence before submission.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Toxicologist correctly approved a submission with acceptable visible safety evidence.",
                }
            if action.action_type in {"edit", "restart"}:
                toxicity_delta = simulated_properties["toxicity"] - previous_properties["toxicity"]
                if toxicity_delta > 0.08:
                    return {
                        "type": "objection",
                        "hard_veto": True,
                        "reason": "Toxicologist correctly raised a hard objection to a major safety regression.",
                    }
                if (
                    simulated_properties["toxicity"] > tox_threshold + 0.02
                    and toxicity_delta >= -0.02
                ):
                    return {
                        "type": "objection",
                        "hard_veto": True,
                        "reason": "Toxicologist correctly blocked a move that left an unsafe scaffold unimproved.",
                    }
                if simulated_properties["toxicity"] > tox_threshold + 0.02:
                    return {
                        "type": "approval",
                        "hard_veto": False,
                        "reason": "Toxicologist correctly allowed a risk-reducing move while residual safety work remains.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Toxicologist correctly approved a safety-compatible move.",
                }
            return neutral

        if role == "assay_planner":
            if action.action_type == "run_assay":
                info_gain = self._estimate_information_gain(action.tool_name or "")
                prior_runs = self._assay_runs.get(f"{current_signature}::{action.tool_name}", 0)
                if (action.tool_name == "run_md_simulation" and self._state.remaining_budget < 4500) or (
                    prior_runs > 0 and info_gain < 0.05
                ):
                    return {
                        "type": "rejection",
                        "hard_veto": True,
                        "reason": "Assay Planner correctly blocked a wasteful or over-expensive assay.",
                    }
                if info_gain < 0.04 and action.tool_name != "search_literature":
                    return {
                        "type": "rejection",
                        "hard_veto": False,
                        "reason": "Assay Planner correctly questioned a low-value assay.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Assay Planner correctly approved an information-efficient assay.",
                }
            if action.action_type == "submit":
                required_props = ["potency", "toxicity"]
                if "synth_min" in self._scenario.hard_constraints:
                    required_props.append("synth")
                missing = [
                    prop for prop in required_props if self._current_property_estimate(prop, current_signature) is None
                ]
                if missing:
                    return {
                        "type": "assay_request",
                        "hard_veto": True,
                        "reason": "Assay Planner correctly asked for missing evidence before submission.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Assay Planner correctly approved a well-supported submission.",
                }
            if action.action_type == "restart":
                potency_threshold = self._scenario.hard_constraints.get("potency_min", 0.72)
                if self._scenario.trap_penalty and previous_properties["potency"] < potency_threshold:
                    return {
                        "type": "approval",
                        "hard_veto": False,
                        "reason": "Assay Planner correctly endorsed escaping a low-value scaffold family.",
                    }
                return {
                    "type": "rejection",
                    "hard_veto": False,
                    "reason": "Assay Planner correctly questioned an unnecessary restart.",
                }
            if action.action_type == "edit":
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Assay Planner correctly approved a low-cost edit before spending assay budget.",
                }
            return neutral

        if role == "process_chemist":
            if action.action_type == "run_assay":
                if action.tool_name == "estimate_synthesizability":
                    return {
                        "type": "approval",
                        "hard_veto": False,
                        "reason": "Process Chemist correctly requested explicit synthesizeability evidence.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Process Chemist correctly allowed an assay that does not worsen route feasibility.",
                }
            if action.action_type == "submit":
                if known_synth is not None and known_synth < synth_threshold:
                    return {
                        "type": "objection",
                        "hard_veto": simulated_properties["synth"] < synth_threshold,
                        "reason": "Process Chemist correctly objected to a submission with visible route evidence below threshold.",
                    }
                if simulated_properties["synth"] < synth_threshold:
                    return {
                        "type": "objection",
                        "hard_veto": "synth_min" in self._scenario.hard_constraints,
                        "reason": "Process Chemist correctly blocked a submission that looks infeasible to make.",
                    }
                if known_synth is None:
                    return {
                        "type": "assay_request",
                        "hard_veto": False,
                        "reason": "Process Chemist correctly asked for synthesizeability evidence before submission.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Process Chemist correctly approved a feasible-looking submission.",
                }
            if action.action_type in {"edit", "restart"}:
                if simulated_properties["synth"] < synth_threshold - 0.03:
                    return {
                        "type": "objection",
                        "hard_veto": False,
                        "reason": "Process Chemist correctly flagged a severe feasibility regression.",
                    }
                if previous_properties["synth"] - simulated_properties["synth"] > 0.08:
                    return {
                        "type": "objection",
                        "hard_veto": False,
                        "reason": "Process Chemist correctly objected to a less tractable route.",
                    }
                return {
                    "type": "approval",
                    "hard_veto": False,
                    "reason": "Process Chemist correctly approved a tractable chemistry move.",
                }
            return neutral

        return neutral

    @staticmethod
    def _matches_feedback(actual_type: str, expected_type: str) -> bool:
        if expected_type == "neutral":
            return False
        if expected_type == "approval":
            return actual_type in {"approval", "submission_recommendation"}
        if expected_type == "objection":
            return actual_type in {"objection", "risk_flag", "rejection"}
        if expected_type == "rejection":
            return actual_type in {"rejection", "objection"}
        if expected_type == "assay_request":
            return actual_type == "assay_request"
        return actual_type == expected_type

    def _evaluate_reasoning_consistency(
        self,
        action: MolForgeAction,
        previous_properties: Mapping[str, float],
        current_properties: Mapping[str, float],
        reward_components: List[RewardComponent],
    ) -> float:
        del previous_properties, current_properties

        rationale = action.rationale.lower().strip()
        evidence = [item.lower().strip() for item in action.evidence if item.strip()]
        expected_effects = {key: value for key, value in action.expected_effects.items() if value}
        score = 0.0
        explanations = []

        if rationale:
            score += 0.02
            explanations.append("short rationale present")
        else:
            score -= 0.03
            explanations.append("missing rationale")

        if evidence:
            grounded = sum(1 for item in evidence if self._evidence_item_is_visible(item))
            score += min(grounded, 3) * 0.015
            if grounded < len(evidence):
                score -= min(len(evidence) - grounded, 2) * 0.02
            explanations.append(f"{grounded}/{len(evidence)} evidence item(s) matched visible state")
        else:
            score -= 0.03
            explanations.append("missing visible evidence")

        if expected_effects:
            plausible = sum(
                1
                for metric, direction in expected_effects.items()
                if self._expected_effect_is_plausible(action, metric, direction)
            )
            checked = len(expected_effects)
            score += min(plausible, 3) * 0.01
            if plausible < checked:
                score -= min(checked - plausible, 2) * 0.015
            explanations.append(f"{plausible}/{checked} expected effect(s) were directionally plausible")
        else:
            score -= 0.02
            explanations.append("missing expected effects")

        score = max(-0.04, min(0.04, score))

        if score != 0.0:
            reward_components.append(
                RewardComponent(
                    name="reasoning_grounding",
                    value=round(score, 4),
                    explanation="; ".join(explanations),
                )
            )
        return score

    def _evidence_item_is_visible(self, item: str) -> bool:
        if not item:
            return False
        visible_terms = {
            self._scenario.scenario_id.lower(),
            self._scenario.difficulty.lower(),
            self._molecule_signature().lower(),
            str(self._state.remaining_budget),
            str(self._state.max_budget),
            str(self._state.step_count),
            str(self._scenario.max_steps),
        }
        visible_terms.update(fragment.lower() for fragment in self._molecule.values())
        visible_terms.update(tool.lower() for tool in self._scenario.enabled_tools)
        visible_terms.update(constraint.lower() for constraint in self._scenario.hard_constraints)
        visible_terms.update(reading.property_name.lower() for reading in self._known_assays)
        visible_terms.update(reading.tool_name.lower() for reading in self._known_assays)
        return any(term and term in item for term in visible_terms)

    def _expected_effect_is_plausible(
        self,
        action: MolForgeAction,
        metric: str,
        direction: str,
    ) -> bool:
        if metric not in {"potency", "toxicity", "synth", "novelty", "budget"}:
            return False
        if direction not in {"up", "down", "neutral", "unknown", "not_applicable"}:
            return False
        if direction in {"unknown", "not_applicable"}:
            return True
        if metric == "budget":
            if action.action_type in {"run_assay", "restart"}:
                return direction == "down"
            return direction == "neutral"
        if action.action_type in {"run_assay", "submit", "defer"}:
            return direction in {"neutral", "unknown", "not_applicable"}
        if action.action_type == "restart":
            if metric in {"toxicity", "budget"}:
                return direction == "down"
            if metric == "synth":
                return direction in {"up", "unknown"}
            return direction in {"up", "unknown", "neutral"}
        if action.action_type != "edit" or not action.slot or not action.fragment:
            return direction in {"neutral", "unknown"}

        fragment = action.fragment
        plausibility = {
            ("hinge", "azaindole", "potency", "up"),
            ("back_pocket", "cyano", "potency", "up"),
            ("back_pocket", "cyano", "toxicity", "down"),
            ("back_pocket", "chloro", "potency", "up"),
            ("back_pocket", "chloro", "toxicity", "up"),
            ("back_pocket", "trifluoromethyl", "potency", "up"),
            ("back_pocket", "trifluoromethyl", "toxicity", "up"),
            ("solvent_tail", "morpholine", "toxicity", "down"),
            ("solvent_tail", "morpholine", "synth", "up"),
            ("solvent_tail", "dimethylamino", "toxicity", "up"),
            ("warhead", "reversible_cyanoacrylamide", "toxicity", "down"),
            ("warhead", "reversible_cyanoacrylamide", "novelty", "up"),
            ("warhead", "nitrile", "toxicity", "down"),
        }
        if (action.slot, fragment, metric, direction) in plausibility:
            return True
        return direction in {"neutral", "unknown"}
