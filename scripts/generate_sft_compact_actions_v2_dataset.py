"""Generate compact MolForge SFT data with stricter enum teaching.

V2 is designed for the observed compact-model failures:
- no nested messages
- no copied observation keys in assistant output
- edit_type uses "substitute", never "replace"
- evidence is always a list
- expected_effects is always a dict with only enum values
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_common import MolForgeAction, MolForgeObservation, attach_reasoning_fields, heuristic_team_action  # noqa: E402
from models import MolForgeAction as LocalMolForgeAction  # noqa: E402
from scenarios import DEFAULT_TOOL_COSTS, FRAGMENT_LIBRARY  # noqa: E402
from server.molforge_environment import MolForgeEnvironment  # noqa: E402


SYSTEM_PROMPT = """
You output one compact MolForgeAction JSON object.

Return exactly these top-level keys and no others:
action_type, acting_role, edit_type, slot, fragment, tool_name, rationale,
evidence, expected_effects.

Allowed action_type values:
edit, run_assay, submit, restart, defer.

Allowed edit_type values:
add_fragment, substitute, remove, undo_last_edit, null.
Use "substitute" for replacement edits. Never output "replace".

Allowed expected_effects values:
up, down, neutral, unknown, not_applicable.

Required field types:
- evidence is an array of short strings.
- expected_effects is an object with potency, toxicity, synth, novelty, budget.
- unused optional fields are JSON null.
- do not copy observation fields such as remaining_budget, step_index,
  known_assays, tool_costs, evidence_gaps, or candidate_edits into the answer.

The environment attaches governance messages automatically. Do not output
messages or message_type.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact MolForge v2 SFT JSONL.")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--output", default="issue/molforge_sft_compact_actions_v2.jsonl")
    parser.add_argument("--randomized", action="store_true")
    args = parser.parse_args()

    if args.randomized:
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"

    env = MolForgeEnvironment()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    observations: list[MolForgeObservation] = []

    for _ in range(args.episodes):
        observation = env.reset()
        for _ in range(args.max_turns):
            if observation.done:
                break
            observations.append(observation)
            action = heuristic_team_action(observation)
            add_record(records, seen, observation, action, source="policy")
            observation = env.step(action)

    for observation, action, source in coverage_examples(observations[:90]):
        action = attach_reasoning_fields(observation, action)
        add_record(records, seen, observation, action, source=source)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(json.dumps(summarize(records, str(output)), indent=2))


def add_record(
    records: list[dict[str, Any]],
    seen: set[str],
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> None:
    record = make_record(observation, action, source=source)
    key = json.dumps(
        {"user": record["messages"][1]["content"], "assistant": record["messages"][2]["content"]},
        sort_keys=True,
    )
    if key in seen:
        return
    seen.add(key)
    validate_target(record["messages"][2]["content"])
    records.append(record)


def coverage_examples(
    observations: list[MolForgeObservation],
) -> list[tuple[MolForgeObservation, MolForgeAction, str]]:
    examples: list[tuple[MolForgeObservation, MolForgeAction, str]] = []
    for observation in observations:
        current = {slot.slot: slot.fragment for slot in observation.molecule_slots}
        for slot, fragments in FRAGMENT_LIBRARY.items():
            for fragment in fragments:
                if current.get(slot) == fragment:
                    continue
                examples.append(
                    (
                        observation,
                        MolForgeAction(
                            action_type="edit",
                            acting_role="lead_chemist",
                            edit_type="substitute",
                            slot=slot,  # type: ignore[arg-type]
                            fragment=fragment,
                            rationale=f"Use substitute on {slot} to {fragment} and then verify evidence.",
                        ),
                        "coverage_edit_substitute",
                    )
                )
        for slot in FRAGMENT_LIBRARY:
            examples.append(
                (
                    observation,
                    MolForgeAction(
                        action_type="edit",
                        acting_role="lead_chemist",
                        edit_type="remove",
                        slot=slot,  # type: ignore[arg-type]
                        rationale=f"Remove the current {slot} choice to simplify the scaffold.",
                    ),
                    "coverage_edit_remove",
                )
            )
        for tool_name in DEFAULT_TOOL_COSTS:
            examples.append(
                (
                    observation,
                    MolForgeAction(
                        action_type="run_assay",
                        acting_role="assay_planner",
                        tool_name=tool_name,
                        rationale=f"Run {tool_name} to close an evidence gap.",
                    ),
                    "coverage_run_assay",
                )
            )
        if observation.scenario_id == "level_2_hard" and observation.remaining_budget >= 350:
            examples.append(
                (
                    observation,
                    MolForgeAction(
                        action_type="restart",
                        acting_role="lead_chemist",
                        rationale="Restart early in the hard task to leave the trap scaffold.",
                    ),
                    "coverage_restart",
                )
            )
        if should_cover_submit(observation):
            examples.append(
                (
                    observation,
                    MolForgeAction(
                        action_type="submit",
                        acting_role="lead_chemist",
                        rationale="Submit when visible evidence is enough for a final decision.",
                    ),
                    "coverage_submit",
                )
            )
        examples.append(
            (
                observation,
                MolForgeAction(
                    action_type="defer",
                    acting_role="lead_chemist",
                    rationale="Defer because no safe evidence-backed action is available.",
                ),
                "coverage_defer",
            )
        )
    return examples


def should_cover_submit(observation: MolForgeObservation) -> bool:
    known = {
        reading.property_name
        for reading in observation.known_assays
        if reading.molecule_signature == observation.current_molecule
    }
    return len(known & {"potency", "toxicity", "synth"}) >= 2 or observation.step_index >= observation.max_steps - 2


def make_record(
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload(observation), separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(target_action(action), separators=(",", ":"))},
        ],
        "metadata": {
            "source": source,
            "scenario_id": observation.scenario_id,
            "difficulty": observation.difficulty,
            "step_index": observation.step_index,
            "action_type": action.action_type,
        },
    }


def user_payload(observation: MolForgeObservation) -> dict[str, Any]:
    lead_view = next((role.observation for role in observation.role_observations if role.role == "lead_chemist"), {})
    assay_view = next((role.observation for role in observation.role_observations if role.role == "assay_planner"), {})
    return {
        "task": "choose_next_compact_action",
        "answer_keys": [
            "action_type",
            "acting_role",
            "edit_type",
            "slot",
            "fragment",
            "tool_name",
            "rationale",
            "evidence",
            "expected_effects",
        ],
        "allowed_values": {
            "action_type": ["edit", "run_assay", "submit", "restart", "defer"],
            "acting_role": ["lead_chemist", "assay_planner"],
            "edit_type": ["add_fragment", "substitute", "remove", "undo_last_edit", None],
            "effect": ["up", "down", "neutral", "unknown", "not_applicable"],
        },
        "observation": {
            "scenario_id": observation.scenario_id,
            "difficulty": observation.difficulty,
            "task_brief": observation.task_brief,
            "state_label": observation.state_label,
            "current_molecule": observation.current_molecule,
            "current_smiles": observation.metadata.get("current_smiles", ""),
            "visible_metrics": observation.visible_metrics,
            "constraints": [constraint.model_dump() for constraint in observation.constraint_status],
            "budget": {
                "remaining": observation.remaining_budget,
                "maximum": observation.max_budget,
            },
            "step": {
                "index": observation.step_index,
                "maximum": observation.max_steps,
            },
            "molecule_slots": lead_view.get("molecule_slots", {}),
            "candidate_edits": lead_view.get("candidate_edits", [])[:10],
            "open_questions": lead_view.get("open_questions", []),
            "known_assays": [
                {
                    "tool_name": reading.tool_name,
                    "property_name": reading.property_name,
                    "estimate": reading.estimate,
                    "molecule_signature": reading.molecule_signature,
                }
                for reading in observation.known_assays[-6:]
            ],
            "assay_tools": {
                "costs": assay_view.get("tool_costs", DEFAULT_TOOL_COSTS),
                "evidence_gaps": assay_view.get("evidence_gaps", []),
                "information_value": assay_view.get("estimated_information_value", {}),
            },
        },
    }


def target_action(action: MolForgeAction) -> dict[str, Any]:
    effects = {
        "potency": "unknown",
        "toxicity": "unknown",
        "synth": "unknown",
        "novelty": "unknown",
        "budget": "neutral",
    }
    effects.update({key: value for key, value in action.expected_effects.items() if key in effects})
    return {
        "action_type": action.action_type,
        "acting_role": action.acting_role,
        "edit_type": action.edit_type,
        "slot": action.slot,
        "fragment": action.fragment,
        "tool_name": action.tool_name,
        "rationale": action.rationale[:220],
        "evidence": list(action.evidence[:5]),
        "expected_effects": effects,
    }


def validate_target(text: str) -> None:
    data = json.loads(text)
    allowed = {
        "action_type",
        "acting_role",
        "edit_type",
        "slot",
        "fragment",
        "tool_name",
        "rationale",
        "evidence",
        "expected_effects",
    }
    if set(data) != allowed:
        raise ValueError(f"target keys mismatch: {sorted(data)}")
    if data["action_type"] == "proposal":
        raise ValueError("proposal is not a compact action type")
    if data["edit_type"] == "replace":
        raise ValueError("replace must never be used; use substitute")
    if not isinstance(data["evidence"], list):
        raise ValueError("evidence must be a list")
    if set(data["expected_effects"]) != {"potency", "toxicity", "synth", "novelty", "budget"}:
        raise ValueError("expected_effects must have exactly five keys")
    LocalMolForgeAction(**data)


def summarize(records: list[dict[str, Any]], output: str) -> dict[str, Any]:
    actions: dict[str, int] = {}
    sources: dict[str, int] = {}
    users = set()
    assistants = set()
    for record in records:
        metadata = record["metadata"]
        actions[metadata["action_type"]] = actions.get(metadata["action_type"], 0) + 1
        sources[metadata["source"]] = sources.get(metadata["source"], 0) + 1
        users.add(record["messages"][1]["content"])
        assistants.add(record["messages"][2]["content"])
    return {
        "output": output,
        "records": len(records),
        "unique_user_prompts": len(users),
        "unique_assistant_targets": len(assistants),
        "action_types": actions,
        "sources": sources,
    }


if __name__ == "__main__":
    main()
