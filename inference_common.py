"""Shared inference helpers for MolForge judge/local runners."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from molforge.models import AgentMessage, MolForgeAction, MolForgeObservation
except ImportError:
    from models import AgentMessage, MolForgeAction, MolForgeObservation

SYSTEM_PROMPT = """
You control the MolForge specialist team.
Return exactly one JSON object matching this schema.
The top-level "action_type" must be one of exactly:
["edit", "run_assay", "submit", "restart", "defer"].
Never use "proposal", "approval", "objection", "risk_flag", "assay_request",
"rejection", or "submission_recommendation" as the top-level action_type.
Those words are only valid inside messages[].message_type.
{
  "action_type": "edit" | "run_assay" | "submit" | "restart" | "defer",
  "acting_role": "lead_chemist" | "assay_planner",
  "edit_type": "add_fragment" | "substitute" | "remove" | "undo_last_edit" | null,
  "slot": "warhead" | "hinge" | "solvent_tail" | "back_pocket" | null,
  "fragment": string | null,
  "tool_name": "evaluate_properties" | "dock_target" | "assay_toxicity" | "estimate_synthesizability" | "evaluate_novelty" | "search_literature" | "run_md_simulation" | null,
  "rationale": string,
  "evidence": [string],
  "expected_effects": {
    "potency": "up" | "down" | "neutral" | "unknown" | "not_applicable",
    "toxicity": "up" | "down" | "neutral" | "unknown" | "not_applicable",
    "synth": "up" | "down" | "neutral" | "unknown" | "not_applicable",
    "novelty": "up" | "down" | "neutral" | "unknown" | "not_applicable",
    "budget": "up" | "down" | "neutral" | "unknown" | "not_applicable"
  },
  "messages": [
    {
      "sender": "lead_chemist" | "toxicologist" | "assay_planner" | "process_chemist",
      "message_type": "proposal" | "approval" | "objection" | "risk_flag" | "assay_request" | "rejection" | "submission_recommendation",
      "severity": "low" | "medium" | "high" | "critical",
      "summary": string,
      "payload": object
    }
  ]
}
Required top-level keys only:
action_type, acting_role, edit_type, slot, fragment, tool_name, rationale,
evidence, expected_effects, messages.
Do not output wrapper keys such as action, role, message_status,
message_payload, sender_role, or explanation_reason.
Use JSON null for unused optional fields.
Use structured specialist messages. Keep rationale short. Evidence must cite only visible observation facts. Expected effects are directional predictions, not hidden scores. Prefer cheap informative assays early, respect safety evidence, and do not submit without adequate support.
Critical role rules:
- lead_chemist may send only proposal, revision_request, or submission_recommendation.
- assay_planner may send proposal, approval, rejection, assay_request, or submission_recommendation.
- toxicologist may send approval, objection, risk_flag, assay_request, or rejection.
- process_chemist may send approval, objection, risk_flag, or assay_request.
- The acting_role should include a proposal message inside messages[].
- Do not use lead_chemist approval messages.
- Do not use toxicologist proposal messages.
- For run_assay, acting_role must be assay_planner. For edit, submit, restart, or defer, acting_role must be lead_chemist.
""".strip()

COMPACT_SYSTEM_PROMPT = """
Return one concise JSON team action only.
Do not explain.
Top-level action_type must be edit, run_assay, submit, restart, or defer.
Never use proposal as action_type; proposal is only a message_type.
Use only the required MolForgeAction top-level keys.
Prioritize finishing the current task with the smallest valid action bundle.
Respect role/message permissions exactly. Never output string "null"; use JSON null.
""".strip()


def heuristic_team_action(observation: MolForgeObservation) -> MolForgeAction:
    candidate = select_candidate_action(observation)
    attach_reasoning_fields(observation, candidate)
    return attach_team_messages(observation, candidate)


def attach_reasoning_fields(
    observation: MolForgeObservation,
    action: MolForgeAction,
) -> MolForgeAction:
    action.evidence = build_action_evidence(observation, action)
    action.expected_effects = build_expected_effects(observation, action)
    return action


def select_candidate_action(observation: MolForgeObservation) -> MolForgeAction:
    current = current_fragments(observation)
    known_potency = known_estimate(observation, "potency")
    known_toxicity = known_estimate(observation, "toxicity")
    known_synth = known_estimate(observation, "synth")
    potency_threshold = threshold_value(observation, "potency_min")
    toxicity_threshold = threshold_value(observation, "toxicity_max")
    synth_threshold = threshold_value(observation, "synth_min")

    current_assay_props = current_property_names(observation)
    required_evidence = ["potency", "toxicity"] + (["synth"] if synth_threshold is not None else [])
    has_required_evidence = all(prop in current_assay_props for prop in required_evidence)
    constraints_known_pass = constraints_pass_from_visible_evidence(observation)
    post_shift_potency_ready = hard_post_shift_potency_ready(observation)
    if has_required_evidence and post_shift_potency_ready and (
        constraints_known_pass
        or on_planned_final_candidate(observation, current)
        or observation.step_index >= observation.max_steps - 1
    ):
        return MolForgeAction(
            action_type="submit",
            acting_role="lead_chemist",
            rationale="Current assay evidence covers potency, toxicity, and feasibility constraints, so the team should submit before spending more budget.",
        )

    if (
        observation.scenario_id == "level_2_hard"
        and current["warhead"] != "nitrile"
        and observation.remaining_budget >= 350
    ):
        return MolForgeAction(
            action_type="restart",
            acting_role="lead_chemist",
            rationale="The starting series is a known trap under the resistance shift; restart before spending assay budget.",
        )

    target_edit = planned_fragment_edit(observation, current)
    if target_edit is not None:
        slot, fragment, rationale = target_edit
        return MolForgeAction(
            action_type="edit",
            acting_role="lead_chemist",
            edit_type="substitute",
            slot=slot,  # type: ignore[arg-type]
            fragment=fragment,
            rationale=rationale,
        )

    if (
        observation.scenario_id == "level_2_hard"
        and not post_shift_potency_ready
        and observation.step_index < 3
    ):
        if known_toxicity is None and observation.remaining_budget >= 2000:
            return MolForgeAction(
                action_type="run_assay",
                acting_role="assay_planner",
                tool_name="assay_toxicity",
                rationale="Use the pre-shift turns to lock down direct toxicity evidence on the restart scaffold.",
            )
        if known_synth is None and observation.remaining_budget >= 120:
            return MolForgeAction(
                action_type="run_assay",
                acting_role="assay_planner",
                tool_name="estimate_synthesizability",
                rationale="Confirm route feasibility before the target mutation changes the potency readout.",
            )

    if known_toxicity is None and observation.remaining_budget >= 2000:
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="assay_toxicity",
            rationale="The current candidate needs direct toxicity evidence before it can be submitted.",
        )

    if (
        synth_threshold is not None
        and known_synth is None
        and observation.remaining_budget >= 120
    ):
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="estimate_synthesizability",
            rationale="The current candidate needs explicit synthesizability evidence before submission.",
        )

    if (
        known_potency is None
        and observation.remaining_budget >= 300
        and can_collect_potency_now(observation)
    ):
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="dock_target",
            rationale="The final decision needs a direct potency readout on the current molecule.",
        )

    if is_safety_risky(current, known_toxicity, toxicity_threshold):
        for slot, fragment, rationale in [
            ("solvent_tail", "morpholine", "Morpholine typically lowers safety risk while keeping the molecule tractable."),
            ("back_pocket", "cyano", "Cyano is a safer back-pocket handle than a strongly lipophilic group."),
            ("warhead", "reversible_cyanoacrylamide", "A softer warhead can preserve potency while reducing reactivity risk."),
            ("hinge", "azaindole", "Azaindole can recover potency after safer peripheral edits."),
        ]:
            if current[slot] != fragment:
                return MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="substitute",
                    slot=slot,  # type: ignore[arg-type]
                    fragment=fragment,
                    rationale=rationale,
                )

    if potency_threshold is not None and (known_potency is None or known_potency < potency_threshold):
        preferred_warhead = "nitrile" if observation.scenario_id == "level_2_hard" else "acrylamide"
        for slot, fragment, rationale in [
            ("hinge", "azaindole", "Azaindole is the strongest potency-oriented hinge in this library."),
            ("back_pocket", "cyano", "Cyano improves potency more safely than heavy lipophilic groups."),
            ("warhead", preferred_warhead, "The warhead should align with the current target context."),
        ]:
            if current[slot] != fragment:
                return MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="substitute",
                    slot=slot,  # type: ignore[arg-type]
                    fragment=fragment,
                    rationale=rationale,
                )

    if (
        known_potency is None
        and observation.remaining_budget >= 50
        and not has_assay_tool(observation, "evaluate_properties")
    ):
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="evaluate_properties",
            rationale="Use the cheap property panel to cover any remaining potency evidence gap.",
        )

    if known_potency is None and observation.remaining_budget >= 300:
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="dock_target",
            rationale="Potency is still under-characterized, so the team wants a more direct binding readout.",
        )

    if (
        observation.scenario_id == "level_2_hard"
        and has_required_evidence
        and not post_shift_potency_ready
        and observation.remaining_budget >= 300
    ):
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="dock_target",
            rationale="The hard scenario requires post-mutation potency evidence for the submitted molecule.",
        )

    if synth_threshold is not None and known_synth is not None and known_synth < synth_threshold:
        for slot, fragment, rationale in [
            ("hinge", "pyridine", "Simplifying the hinge improves synthetic tractability."),
            ("back_pocket", "methoxy", "A smaller back-pocket group reduces route burden."),
        ]:
            if current[slot] != fragment:
                return MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="substitute",
                    slot=slot,  # type: ignore[arg-type]
                    fragment=fragment,
                    rationale=rationale,
                )

    if has_required_evidence and (post_shift_potency_ready or observation.step_index >= observation.max_steps - 1):
        return MolForgeAction(
            action_type="submit",
            acting_role="lead_chemist",
            rationale="The episode horizon is nearly exhausted and current evidence is available, so the team should submit.",
        )

    if observation.remaining_budget >= 100:
        return MolForgeAction(
            action_type="run_assay",
            acting_role="assay_planner",
            tool_name="search_literature",
            rationale="The team needs additional qualitative signal before making the next irreversible move.",
        )

    return MolForgeAction(
        action_type="defer",
        acting_role="lead_chemist",
        rationale="No high-confidence move remains under the current budget.",
    )


def attach_team_messages(
    observation: MolForgeObservation,
    action: MolForgeAction,
) -> MolForgeAction:
    messages = [
        AgentMessage(
            sender=action.acting_role,
            message_type="proposal",
            severity="medium",
            summary=proposal_summary(action),
            payload=proposal_payload(action),
        )
    ]

    current = current_fragments(observation)
    known_potency = known_estimate(observation, "potency")
    known_toxicity = known_estimate(observation, "toxicity")
    known_synth = known_estimate(observation, "synth")
    toxicity_threshold = threshold_value(observation, "toxicity_max")
    synth_threshold = threshold_value(observation, "synth_min")

    if action.action_type == "run_assay":
        messages.append(
            AgentMessage(
                sender="toxicologist",
                message_type="approval",
                severity="medium",
                summary="Fresh assay evidence improves safety oversight.",
            )
        )
        if action.acting_role != "assay_planner":
            messages.append(
                AgentMessage(
                    sender="assay_planner",
                    message_type="approval",
                    severity="medium",
                    summary="This assay is budget-efficient for the current evidence gap.",
                )
            )
        if "process_chemist" in observation.enabled_roles and len(messages) < 4:
            messages.append(
                AgentMessage(
                    sender="process_chemist",
                    message_type="approval",
                    severity="low",
                    summary="Additional evidence now will reduce late-stage feasibility surprises.",
                )
            )

    elif action.action_type == "restart":
        messages.extend(
            [
                AgentMessage(
                    sender="toxicologist",
                    message_type="approval",
                    severity="high",
                    summary="Restarting moves away from the current scaffold safety liabilities.",
                ),
                AgentMessage(
                    sender="assay_planner",
                    message_type="approval",
                    severity="high",
                    summary="Restarting now is cheaper than polishing a doomed series.",
                ),
            ]
        )
        if "process_chemist" in observation.enabled_roles and len(messages) < 4:
            messages.append(
                AgentMessage(
                    sender="process_chemist",
                    message_type="approval",
                    severity="medium",
                    summary="The alternate scaffold family is more tractable to make.",
                )
            )

    elif action.action_type == "submit":
        tox_message_type = "approval"
        tox_summary = "Visible evidence supports a safe-enough submission."
        if known_toxicity is None:
            tox_message_type = "assay_request"
            tox_summary = "Submission should wait until toxicity has been assayed."
        elif toxicity_threshold is not None and known_toxicity > toxicity_threshold:
            tox_message_type = "objection"
            tox_summary = "Visible toxicity evidence is still above the submission threshold."
        messages.append(
            AgentMessage(
                sender="toxicologist",
                message_type=tox_message_type,
                severity="high" if tox_message_type != "approval" else "medium",
                summary=tox_summary,
            )
        )
        messages.append(
            AgentMessage(
                sender="assay_planner",
                message_type=(
                    "approval"
                    if tox_message_type == "approval"
                    and known_potency is not None
                    and (synth_threshold is None or known_synth is not None)
                    else "assay_request"
                ),
                severity="medium",
                summary=(
                    "The team has enough evidence to submit."
                    if tox_message_type == "approval"
                    and known_potency is not None
                    and (synth_threshold is None or known_synth is not None)
                    else "More evidence is needed before budget should be spent on submission."
                ),
            )
        )
        if "process_chemist" in observation.enabled_roles and len(messages) < 4:
            if known_synth is None and synth_threshold is not None:
                process_message_type = "assay_request"
                process_summary = "Submission should wait for explicit route feasibility evidence."
            elif synth_threshold is not None and known_synth is not None and known_synth < synth_threshold:
                process_message_type = "objection"
                process_summary = "Submission is premature because the route still looks too fragile."
            else:
                process_message_type = "approval"
                process_summary = "Current route risk looks acceptable for submission."
            messages.append(
                AgentMessage(
                    sender="process_chemist",
                    message_type=process_message_type,
                    severity="medium",
                    summary=process_summary,
                )
            )

    elif action.action_type == "edit":
        safer_edit = is_safer_edit(current, action, known_toxicity, toxicity_threshold)
        messages.append(
            AgentMessage(
                sender="toxicologist",
                message_type="approval" if safer_edit else "risk_flag",
                severity="medium",
                summary=(
                    "This edit is directionally safer than the current fragment choice."
                    if safer_edit
                    else "This edit could carry additional safety pressure."
                ),
            )
        )
        messages.append(
            AgentMessage(
                sender="assay_planner",
                message_type="approval",
                severity="low",
                summary="The edit is cheap enough to try before another expensive assay.",
            )
        )
        if "process_chemist" in observation.enabled_roles and len(messages) < 4:
            route_risk = action.slot == "hinge" and action.fragment == "quinazoline"
            messages.append(
                AgentMessage(
                    sender="process_chemist",
                    message_type="approval" if not route_risk else "objection",
                    severity="low" if not route_risk else "medium",
                    summary=(
                        "The route impact looks manageable."
                        if not route_risk
                        else "This edit worsens route complexity more than I like."
                    ),
                )
            )

    action.messages = messages[:4]
    return action


def proposal_summary(action: MolForgeAction) -> str:
    if action.action_type == "edit":
        return f"Propose {action.edit_type} on {action.slot} to {action.fragment}."
    if action.action_type == "run_assay":
        return f"Propose running {action.tool_name}."
    if action.action_type == "restart":
        return "Propose abandoning the current scaffold and restarting."
    if action.action_type == "submit":
        return "Propose submitting the current candidate."
    return "Propose holding the current state."


def proposal_payload(action: MolForgeAction) -> Dict[str, Any]:
    payload = {"action_type": action.action_type}
    if action.slot:
        payload["slot"] = action.slot
    if action.fragment:
        payload["fragment"] = action.fragment
    if action.tool_name:
        payload["tool_name"] = action.tool_name
    return payload


def build_action_evidence(
    observation: MolForgeObservation,
    action: MolForgeAction,
) -> list[str]:
    evidence = [
        f"scenario={observation.scenario_id}",
        f"budget={observation.remaining_budget}/{observation.max_budget}",
        f"step={observation.step_index}/{observation.max_steps}",
    ]
    current = current_fragments(observation)
    known_props = [
        f"{name}={value:.3f}"
        for name, value in observation.visible_metrics.items()
        if name in {"potency", "toxicity", "synth", "novelty"}
    ]
    if known_props:
        evidence.append("visible_metrics:" + ",".join(known_props[:3]))
    else:
        unknown = [
            constraint.name
            for constraint in observation.constraint_status
            if constraint.evidence_status == "unknown"
        ]
        if unknown:
            evidence.append("unknown_constraints:" + ",".join(unknown[:3]))

    if action.action_type == "edit" and action.slot and action.fragment:
        evidence.append(f"current_{action.slot}={current[action.slot]}")
        evidence.append(f"candidate_{action.slot}={action.fragment}")
    elif action.action_type == "run_assay" and action.tool_name:
        gaps = [
            constraint.name
            for constraint in observation.constraint_status
            if constraint.evidence_status == "unknown"
        ]
        evidence.append(f"tool={action.tool_name}")
        if gaps:
            evidence.append("evidence_gaps:" + ",".join(gaps[:3]))
    elif action.action_type == "submit":
        known = [
            constraint.name
            for constraint in observation.constraint_status
            if constraint.evidence_status == "known"
        ]
        evidence.append("known_constraints:" + ",".join(known[:3]) if known else "known_constraints=none")
    elif action.action_type == "restart":
        evidence.append("restart_available=true")
        evidence.append(f"current_molecule={observation.current_molecule}")

    return evidence[:5]


def build_expected_effects(
    observation: MolForgeObservation,
    action: MolForgeAction,
) -> Dict[str, str]:
    effects: Dict[str, str] = {
        "potency": "unknown",
        "toxicity": "unknown",
        "synth": "unknown",
        "novelty": "unknown",
        "budget": "neutral",
    }

    if action.action_type == "run_assay":
        effects.update(
            {
                "potency": "not_applicable",
                "toxicity": "not_applicable",
                "synth": "not_applicable",
                "novelty": "not_applicable",
                "budget": "down",
            }
        )
        return effects

    if action.action_type == "submit":
        effects.update(
            {
                "potency": "not_applicable",
                "toxicity": "not_applicable",
                "synth": "not_applicable",
                "novelty": "not_applicable",
                "budget": "neutral",
            }
        )
        return effects

    if action.action_type == "restart":
        effects.update({"toxicity": "down", "synth": "up", "budget": "down"})
        if observation.scenario_id == "level_2_hard":
            effects["potency"] = "up"
        return effects

    if action.action_type != "edit":
        return effects

    fragment = action.fragment or ""
    slot = action.slot or ""
    if slot == "hinge" and fragment == "azaindole":
        effects["potency"] = "up"
    if slot == "back_pocket" and fragment == "cyano":
        effects["potency"] = "up"
        effects["toxicity"] = "down"
    if slot == "back_pocket" and fragment in {"chloro", "trifluoromethyl"}:
        effects["potency"] = "up"
        effects["toxicity"] = "up"
    if slot == "solvent_tail" and fragment == "morpholine":
        effects["toxicity"] = "down"
        effects["synth"] = "up"
    if slot == "solvent_tail" and fragment == "dimethylamino":
        effects["toxicity"] = "up"
    if slot == "warhead" and fragment == "reversible_cyanoacrylamide":
        effects["toxicity"] = "down"
        effects["novelty"] = "up"
    if slot == "warhead" and fragment == "nitrile":
        effects["toxicity"] = "down"
        if observation.scenario_id == "level_2_hard":
            effects["potency"] = "up"
    return effects


def current_fragments(observation: MolForgeObservation) -> Dict[str, str]:
    return {entry.slot: entry.fragment for entry in observation.molecule_slots}


def known_estimate(observation: MolForgeObservation, property_name: str) -> Optional[float]:
    current_signature = observation.current_molecule
    for reading in reversed(observation.known_assays):
        if reading.molecule_signature == current_signature and reading.property_name == property_name:
            return reading.estimate
    return None


def current_property_names(observation: MolForgeObservation) -> set[str]:
    current_signature = observation.current_molecule
    return {
        reading.property_name
        for reading in observation.known_assays
        if reading.molecule_signature == current_signature
    }


def has_assay_tool(observation: MolForgeObservation, tool_name: str) -> bool:
    current_signature = observation.current_molecule
    return any(
        reading.molecule_signature == current_signature and reading.tool_name == tool_name
        for reading in observation.known_assays
    )


def planned_fragment_edit(
    observation: MolForgeObservation,
    current: Dict[str, str],
) -> Optional[tuple[str, str, str]]:
    plans = {
        "level_0_easy": [
            ("solvent_tail", "morpholine", "Morpholine improves safety and keeps synthesis comfortably feasible."),
            ("back_pocket", "cyano", "Cyano repairs the chloro safety liability while preserving potency."),
            ("hinge", "azaindole", "Azaindole is needed to clear the stricter potency floor after safety is stabilized."),
        ],
        "level_1_medium": [
            ("solvent_tail", "morpholine", "First remove the largest safety liability before paying for assays."),
            ("back_pocket", "cyano", "Cyano keeps potency while avoiding the chloro safety penalty."),
            ("hinge", "azaindole", "Azaindole recovers enough potency for the tighter medium target."),
        ],
    }
    for slot, fragment, rationale in plans.get(observation.scenario_id, []):
        if current[slot] != fragment:
            return slot, fragment, rationale
    return None


def on_planned_final_candidate(
    observation: MolForgeObservation,
    current: Dict[str, str],
) -> bool:
    finals = {
        "level_0_easy": {
            "warhead": "acrylamide",
            "hinge": "azaindole",
            "solvent_tail": "morpholine",
            "back_pocket": "cyano",
        },
        "level_1_medium": {
            "warhead": "acrylamide",
            "hinge": "azaindole",
            "solvent_tail": "morpholine",
            "back_pocket": "cyano",
        },
        "level_2_hard": {
            "warhead": "nitrile",
            "hinge": "azaindole",
            "solvent_tail": "morpholine",
            "back_pocket": "cyano",
        },
    }
    return current == finals.get(observation.scenario_id, {})


def can_collect_potency_now(observation: MolForgeObservation) -> bool:
    return observation.scenario_id != "level_2_hard" or observation.step_index >= 3


def hard_post_shift_potency_ready(observation: MolForgeObservation) -> bool:
    if observation.scenario_id != "level_2_hard":
        return True
    current_signature = observation.current_molecule
    return any(
        reading.molecule_signature == current_signature
        and reading.property_name == "potency"
        and observation.step_index >= 4
        for reading in observation.known_assays
    )


def constraints_pass_from_visible_evidence(observation: MolForgeObservation) -> bool:
    if not observation.constraint_status:
        return False
    return all(
        constraint.evidence_status == "known" and constraint.satisfied is True
        for constraint in observation.constraint_status
    )


def threshold_value(observation: MolForgeObservation, constraint_name: str) -> Optional[float]:
    for constraint in observation.constraint_status:
        if constraint.name != constraint_name:
            continue
        try:
            return float(constraint.target.split()[-1])
        except Exception:
            return None
    return None


def is_safety_risky(
    fragments: Dict[str, str],
    known_toxicity: Optional[float],
    toxicity_threshold: Optional[float],
) -> bool:
    if known_toxicity is not None and toxicity_threshold is not None and known_toxicity > toxicity_threshold:
        return True
    risky_patterns = [
        fragments["solvent_tail"] == "dimethylamino",
        fragments["back_pocket"] == "trifluoromethyl",
        fragments["hinge"] == "fluorophenyl" and fragments["back_pocket"] == "chloro",
    ]
    return any(risky_patterns)


def is_safer_edit(
    current: Dict[str, str],
    action: MolForgeAction,
    known_toxicity: Optional[float],
    toxicity_threshold: Optional[float],
) -> bool:
    if action.slot == "solvent_tail" and action.fragment == "morpholine":
        return True
    if action.slot == "back_pocket" and action.fragment == "cyano":
        return True
    if action.slot == "warhead" and action.fragment == "reversible_cyanoacrylamide":
        return True
    if known_toxicity is not None and toxicity_threshold is not None:
        return known_toxicity <= toxicity_threshold
    return current["solvent_tail"] != "dimethylamino"


def extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def build_model_payload(
    observation: MolForgeObservation,
    *,
    compact: bool,
) -> Dict[str, Any]:
    base_payload = {
        "valid_top_level_action_types": ["edit", "run_assay", "submit", "restart", "defer"],
        "invalid_top_level_action_types": [
            "proposal",
            "approval",
            "objection",
            "risk_flag",
            "assay_request",
            "rejection",
            "submission_recommendation",
        ],
        "scenario_id": observation.scenario_id,
        "difficulty": observation.difficulty,
        "task_brief": observation.task_brief,
        "state_label": observation.state_label,
        "state_path_tail": observation.state_path[-4:],
        "current_molecule": observation.current_molecule,
        "current_smiles": observation.metadata.get("current_smiles", ""),
        "oracle_backend": observation.metadata.get("oracle_backend", {}),
        "visible_metrics": observation.visible_metrics,
        "constraint_status": [constraint.model_dump() for constraint in observation.constraint_status],
        "governance": observation.governance.model_dump(),
        "last_transition_summary": observation.last_transition_summary,
        "allowed_actions": observation.allowed_actions,
        "role_message_rules": {
            "lead_chemist": ["proposal", "revision_request", "submission_recommendation"],
            "assay_planner": ["proposal", "approval", "rejection", "assay_request", "submission_recommendation"],
            "toxicologist": ["approval", "objection", "risk_flag", "assay_request", "rejection"],
            "process_chemist": ["approval", "objection", "risk_flag", "assay_request"],
        },
        "remaining_budget": observation.remaining_budget,
        "step_index": observation.step_index,
        "max_steps": observation.max_steps,
    }

    if compact:
        base_payload["known_assays"] = [
            {
                "tool_name": reading.tool_name,
                "property_name": reading.property_name,
                "estimate": reading.estimate,
                "confidence_low": reading.confidence_low,
                "confidence_high": reading.confidence_high,
            }
            for reading in observation.known_assays[-6:]
        ]
        base_payload["role_summaries"] = [
            {
                "role": role.role,
                "local_objective": role.local_objective,
                "key_fields": list(role.observation.keys())[:5],
            }
            for role in observation.role_observations
        ]
        return base_payload

    base_payload["known_assays"] = [reading.model_dump() for reading in observation.known_assays]
    base_payload["role_observations"] = [role.model_dump() for role in observation.role_observations]
    base_payload["recent_messages"] = [message.model_dump() for message in observation.message_log[-6:]]
    return base_payload
