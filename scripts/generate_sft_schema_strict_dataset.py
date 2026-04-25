"""Generate a schema-first MolForge SFT dataset.

This dataset is stricter than the policy trace dataset:
- every assistant answer contains exactly the intended MolForgeAction keys
- optional action fields are explicit JSON null when unused
- message objects contain only the keys shown in the prompt
- rare action types, edit types, assay tools, and reviewer message types are
  deliberately over-sampled so small models learn the schema before RL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_common import (  # noqa: E402
    SYSTEM_PROMPT,
    AgentMessage,
    MolForgeAction,
    MolForgeObservation,
    attach_reasoning_fields,
    attach_team_messages,
    build_model_payload,
    heuristic_team_action,
)
from scenarios import DEFAULT_TOOL_COSTS, FRAGMENT_LIBRARY  # noqa: E402
from server.molforge_environment import MolForgeEnvironment  # noqa: E402


MESSAGE_VARIANTS = {
    "toxicologist": ["approval", "risk_flag", "objection", "assay_request", "rejection"],
    "assay_planner": ["approval", "rejection", "assay_request", "submission_recommendation"],
    "process_chemist": ["approval", "risk_flag", "objection", "assay_request"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict MolForge schema SFT JSONL.")
    parser.add_argument("--episodes", type=int, default=75)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--output", default="data/molforge_sft_schema_strict.jsonl")
    parser.add_argument(
        "--randomized",
        action="store_true",
        help="Enable MolForge training randomization while collecting policy traces.",
    )
    args = parser.parse_args()

    if args.randomized:
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = MolForgeEnvironment()
    records: list[dict[str, Any]] = []

    for _ in range(args.episodes):
        observation = env.reset()
        for _ in range(args.max_turns):
            if observation.done:
                break
            action = heuristic_team_action(observation)
            records.append(make_record(observation, action, source="policy_trace"))
            observation = env.step(action)

    for observation, action, source in schema_curriculum_examples():
        action = attach_reasoning_fields(observation, action)
        action.messages = curriculum_messages(observation, action)
        records.append(make_record(observation, action, source=source))

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "records": len(records),
                "curriculum_records": sum(
                    1 for record in records if record["metadata"]["source"] != "policy_trace"
                ),
            },
            indent=2,
        )
    )


def schema_curriculum_examples() -> Iterable[tuple[MolForgeObservation, MolForgeAction, str]]:
    env = MolForgeEnvironment()
    observations = [env.reset(), env.reset(), env.reset()]

    for observation in observations:
        current = {slot.slot: slot.fragment for slot in observation.molecule_slots}
        for slot, fragments in FRAGMENT_LIBRARY.items():
            for fragment in fragments:
                if current[slot] == fragment:
                    continue
                yield observation, MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="substitute",
                    slot=slot,  # type: ignore[arg-type]
                    fragment=fragment,
                    rationale=f"Substitute {slot} to {fragment} and then verify the visible evidence.",
                ), "schema_edit_substitute"
                yield observation, MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="add_fragment",
                    slot=slot,  # type: ignore[arg-type]
                    fragment=fragment,
                    rationale=f"Add a valid {slot} fragment while keeping the action fields exact.",
                ), "schema_edit_add_fragment"

        for slot in FRAGMENT_LIBRARY:
            yield observation, MolForgeAction(
                action_type="edit",
                acting_role="lead_chemist",
                edit_type="remove",
                slot=slot,  # type: ignore[arg-type]
                rationale=f"Remove the current {slot} choice and simplify the scaffold.",
            ), "schema_edit_remove"
            yield observation, MolForgeAction(
                action_type="edit",
                acting_role="lead_chemist",
                edit_type="undo_last_edit",
                slot=slot,  # type: ignore[arg-type]
                rationale=f"Undo the last {slot} edit because visible evidence is not convincing.",
            ), "schema_edit_undo"

        for tool_name in DEFAULT_TOOL_COSTS:
            for _ in range(6):
                yield observation, MolForgeAction(
                    action_type="run_assay",
                    acting_role="assay_planner",
                    tool_name=tool_name,
                    rationale=f"Run {tool_name} to gather evidence before another chemistry decision.",
                ), "schema_run_assay"

        for _ in range(8):
            yield observation, MolForgeAction(
                action_type="defer",
                acting_role="lead_chemist",
                rationale="Defer because no safe evidence-backed action is available this turn.",
            ), "schema_defer"

    hard = observations[-1]
    for _ in range(12):
        yield hard, MolForgeAction(
            action_type="restart",
            acting_role="lead_chemist",
            rationale="Restart early to leave the trap scaffold before spending assay budget.",
        ), "schema_restart"

    submit_observations = collect_submit_observations(limit=24)
    for observation in submit_observations:
        action = MolForgeAction(
            action_type="submit",
            acting_role="lead_chemist",
            rationale="Submit now because visible assay evidence is sufficient for a final decision.",
        )
        action = attach_reasoning_fields(observation, action)
        action = attach_team_messages(observation, action)
        yield observation, action, "schema_submit"


def collect_submit_observations(*, limit: int) -> list[MolForgeObservation]:
    env = MolForgeEnvironment()
    observations: list[MolForgeObservation] = []
    for _ in range(limit * 2):
        observation = env.reset()
        for _ in range(10):
            if observation.done:
                break
            action = heuristic_team_action(observation)
            if action.action_type == "submit":
                observations.append(observation)
                break
            observation = env.step(action)
        if len(observations) >= limit:
            break
    return observations[:limit]


def curriculum_messages(
    observation: MolForgeObservation,
    action: MolForgeAction,
) -> list[AgentMessage]:
    if action.action_type == "defer":
        return [
            AgentMessage(
                sender="lead_chemist",
                message_type="revision_request",
                severity="low",
                summary="Hold the turn until a valid evidence-backed action is clear.",
                payload={"action_type": "defer"},
            )
        ]

    messages = [
        AgentMessage(
            sender=action.acting_role,
            message_type="proposal",
            severity="medium",
            summary=proposal_summary(action),
            payload=proposal_payload(action),
        )
    ]

    reviewers = [
        role
        for role in observation.enabled_roles
        if role != action.acting_role and role in MESSAGE_VARIANTS
    ]
    seed = observation.step_index + len(action.action_type) + len(action.rationale)
    for index, role in enumerate(reviewers[:3]):
        variants = MESSAGE_VARIANTS.get(role, ["approval"])
        message_type = variants[(seed + index) % len(variants)]
        messages.append(
            AgentMessage(
                sender=role,  # type: ignore[arg-type]
                message_type=message_type,  # type: ignore[arg-type]
                severity="medium" if message_type in {"approval", "assay_request"} else "high",
                summary=review_summary(role, message_type, action),
                payload={"action_type": action.action_type},
            )
        )
    return messages


def proposal_summary(action: MolForgeAction) -> str:
    if action.action_type == "edit":
        return f"Propose {action.edit_type} for {action.slot}."
    if action.action_type == "run_assay":
        return f"Propose running {action.tool_name}."
    if action.action_type == "restart":
        return "Propose restarting from the alternate scaffold."
    if action.action_type == "submit":
        return "Propose final submission."
    return "Propose deferring the turn."


def proposal_payload(action: MolForgeAction) -> dict[str, Any]:
    payload: dict[str, Any] = {"action_type": action.action_type}
    if action.edit_type is not None:
        payload["edit_type"] = action.edit_type
    if action.slot is not None:
        payload["slot"] = action.slot
    if action.fragment is not None:
        payload["fragment"] = action.fragment
    if action.tool_name is not None:
        payload["tool_name"] = action.tool_name
    return payload


def review_summary(role: str, message_type: str, action: MolForgeAction) -> str:
    if message_type == "approval":
        return f"{role} approves the proposed {action.action_type}."
    if message_type == "assay_request":
        return f"{role} asks for assay evidence before relying on this decision."
    if message_type == "submission_recommendation":
        return "Assay Planner recommends submission only after evidence is current."
    if message_type == "rejection":
        return f"{role} rejects moving ahead without cleaner support."
    if message_type == "objection":
        return f"{role} objects because this decision may violate constraints."
    return f"{role} flags risk for the proposed {action.action_type}."


def make_record(
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    build_model_payload(observation, compact=False),
                    separators=(",", ":"),
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(canonical_action_dict(action), separators=(",", ":")),
            },
        ],
        "metadata": {
            "source": source,
            "scenario_id": observation.scenario_id,
            "difficulty": observation.difficulty,
            "step_index": observation.step_index,
            "action_type": action.action_type,
        },
    }


def canonical_action_dict(action: MolForgeAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type,
        "acting_role": action.acting_role,
        "edit_type": action.edit_type,
        "slot": action.slot,
        "fragment": action.fragment,
        "tool_name": action.tool_name,
        "rationale": action.rationale[:360],
        "evidence": action.evidence[:5],
        "expected_effects": action.expected_effects,
        "messages": [
            {
                "sender": message.sender,
                "message_type": message.message_type,
                "severity": message.severity,
                "summary": message.summary[:220],
                "payload": message.payload,
            }
            for message in action.messages[:4]
        ],
    }


if __name__ == "__main__":
    main()
