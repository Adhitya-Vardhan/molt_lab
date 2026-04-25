"""Generate compact action-only MolForge SFT data.

The assistant target intentionally omits `messages` so small models do not
confuse nested `message_type="proposal"` with top-level `action_type`.
The runner can attach governance messages deterministically after parsing.
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

from inference_common import (  # noqa: E402
    MolForgeAction,
    MolForgeObservation,
    attach_reasoning_fields,
    heuristic_team_action,
)
from models import MolForgeAction as LocalMolForgeAction  # noqa: E402
from scenarios import DEFAULT_TOOL_COSTS, FRAGMENT_LIBRARY  # noqa: E402
from server.molforge_environment import MolForgeEnvironment  # noqa: E402


COMPACT_ACTION_SYSTEM_PROMPT = """
You control the MolForge action policy.
Return exactly one JSON object with only these top-level keys:
action_type, acting_role, edit_type, slot, fragment, tool_name, rationale,
evidence, expected_effects.

Valid action_type values are exactly:
edit, run_assay, submit, restart, defer.

Do not output team messages. Do not output proposal, approval, objection,
risk_flag, assay_request, rejection, or submission_recommendation as action_type.
The environment will attach governance messages automatically.

Role rules:
- run_assay uses acting_role "assay_planner" and a valid tool_name.
- edit, submit, restart, and defer use acting_role "lead_chemist".
- unused optional fields must be JSON null.
- rationale must be short.
- evidence must cite only visible observation facts.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact MolForge action SFT JSONL.")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--output", default="issue/molforge_sft_compact_actions.jsonl")
    parser.add_argument("--randomized", action="store_true")
    args = parser.parse_args()

    if args.randomized:
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"

    env = MolForgeEnvironment()
    records: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    observations: list[MolForgeObservation] = []

    for _ in range(args.episodes):
        observation = env.reset()
        for _ in range(args.max_turns):
            if observation.done:
                break
            observations.append(observation)
            action = heuristic_team_action(observation)
            add_record(records, seen_records, observation, action, source="policy_trace")
            observation = env.step(action)

    for observation, action, source in coverage_examples(observations[:60]):
        action = attach_reasoning_fields(observation, action)
        add_record(records, seen_records, observation, action, source=source)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(json.dumps(summarize(output, records), indent=2))


def add_record(
    records: list[dict[str, Any]],
    seen_records: set[str],
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> None:
    record = make_record(observation, action, source=source)
    assistant = record["messages"][-1]["content"]
    key = json.dumps(
        {
            "user": record["messages"][1]["content"],
            "assistant": assistant,
        },
        sort_keys=True,
    )
    if key in seen_records:
        return
    seen_records.add(key)
    validate_compact_assistant(assistant)
    records.append(record)


def coverage_examples(
    observations: list[MolForgeObservation],
) -> list[tuple[MolForgeObservation, MolForgeAction, str]]:
    examples: list[tuple[MolForgeObservation, MolForgeAction, str]] = []
    if not observations:
        return examples

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
                            rationale=f"Substitute {slot} to {fragment} and verify the visible evidence.",
                        ),
                        "coverage_edit_substitute",
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
                        rationale=f"Run {tool_name} to close a visible evidence gap.",
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
        known_props = {
            reading.property_name
            for reading in observation.known_assays
            if reading.molecule_signature == observation.current_molecule
        }
        if len(known_props & {"potency", "toxicity", "synth"}) >= 2 or observation.step_index >= observation.max_steps - 2:
            examples.append(
                (
                    observation,
                    MolForgeAction(
                        action_type="submit",
                        acting_role="lead_chemist",
                        rationale="Submit when visible evidence is sufficient and the episode should finish.",
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
                    rationale="Defer because no safe evidence-backed move is available this turn.",
                ),
                "coverage_defer",
            )
        )
    return examples


def make_record(
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": COMPACT_ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(compact_payload(observation), separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": json.dumps(compact_action_dict(action), separators=(",", ":")),
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


def compact_payload(observation: MolForgeObservation) -> dict[str, Any]:
    lead_view = next(
        (role.observation for role in observation.role_observations if role.role == "lead_chemist"),
        {},
    )
    assay_view = next(
        (role.observation for role in observation.role_observations if role.role == "assay_planner"),
        {},
    )
    process_view = next(
        (role.observation for role in observation.role_observations if role.role == "process_chemist"),
        {},
    )
    toxicology_view = next(
        (role.observation for role in observation.role_observations if role.role == "toxicologist"),
        {},
    )
    return {
        "valid_action_types": ["edit", "run_assay", "submit", "restart", "defer"],
        "scenario_id": observation.scenario_id,
        "difficulty": observation.difficulty,
        "task_brief": observation.task_brief,
        "state_label": observation.state_label,
        "current_molecule": observation.current_molecule,
        "current_smiles": observation.metadata.get("current_smiles", ""),
        "visible_metrics": observation.visible_metrics,
        "constraint_status": [constraint.model_dump() for constraint in observation.constraint_status],
        "remaining_budget": observation.remaining_budget,
        "max_budget": observation.max_budget,
        "step_index": observation.step_index,
        "max_steps": observation.max_steps,
        "molecule_slots": lead_view.get("molecule_slots", {}),
        "candidate_edits": lead_view.get("candidate_edits", [])[:12],
        "open_questions": lead_view.get("open_questions", []),
        "known_assays": [
            {
                "tool_name": reading.tool_name,
                "property_name": reading.property_name,
                "estimate": reading.estimate,
                "confidence_low": reading.confidence_low,
                "confidence_high": reading.confidence_high,
                "molecule_signature": reading.molecule_signature,
            }
            for reading in observation.known_assays[-8:]
        ],
        "tool_costs": assay_view.get("tool_costs", DEFAULT_TOOL_COSTS),
        "tool_usage_history": assay_view.get("tool_usage_history", {}),
        "evidence_gaps": assay_view.get("evidence_gaps", []),
        "estimated_information_value": assay_view.get("estimated_information_value", {}),
        "toxicity_threshold": toxicology_view.get("hard_threshold"),
        "safety_alerts": toxicology_view.get("safety_alerts", []),
        "route_warnings": process_view.get("route_warnings", []),
        "feasibility_flags": process_view.get("feasibility_flags", {}),
    }


def compact_action_dict(action: MolForgeAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type,
        "acting_role": action.acting_role,
        "edit_type": action.edit_type,
        "slot": action.slot,
        "fragment": action.fragment,
        "tool_name": action.tool_name,
        "rationale": action.rationale[:260],
        "evidence": action.evidence[:5],
        "expected_effects": action.expected_effects,
    }


def validate_compact_assistant(assistant: str) -> None:
    data = json.loads(assistant)
    if "messages" in data:
        raise ValueError("compact assistant target must not include messages")
    allowed_keys = {
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
    extra = set(data) - allowed_keys
    if extra:
        raise ValueError(f"unexpected compact target keys: {sorted(extra)}")
    LocalMolForgeAction(**data)


def summarize(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    unique_users = set()
    unique_assistants = set()
    for record in records:
        metadata = record["metadata"]
        action_counts[metadata["action_type"]] = action_counts.get(metadata["action_type"], 0) + 1
        sources[metadata["source"]] = sources.get(metadata["source"], 0) + 1
        unique_users.add(record["messages"][1]["content"])
        unique_assistants.add(record["messages"][-1]["content"])
    return {
        "output": str(output),
        "records": len(records),
        "unique_user_prompts": len(unique_users),
        "unique_assistant_targets": len(unique_assistants),
        "action_types": action_counts,
        "sources": sources,
    }


if __name__ == "__main__":
    main()
