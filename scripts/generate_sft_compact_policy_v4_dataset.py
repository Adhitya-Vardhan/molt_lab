"""Generate MolForge compact-policy SFT data aligned to MLX inference.

V4 is designed around the failures seen in the v3 adapter:
- train on the exact compact prompt/payload shape used at inference time
- emphasize successful end-to-end expert trajectories
- include recovery examples after governance vetoes
- include enough schema coverage for all core action types without making
  unsafe edits or wasteful assays dominate the positive training signal
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
    MolForgeAction,
    MolForgeObservation,
    attach_reasoning_fields,
    attach_team_messages,
    heuristic_team_action,
)
from scenarios import DEFAULT_TOOL_COSTS  # noqa: E402
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
- Do not repeat the same assay on an unchanged molecule; edit, restart, or wait for a target-context shift before reassaying.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact MolForge v4 policy SFT JSONL.")
    parser.add_argument("--episodes", type=int, default=520)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", default="policy-v4")
    parser.add_argument("--output", default="issue/molforge_sft_compact_policy_v4.jsonl")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    add_expert_traces(records, seen, episodes=18, max_turns=args.max_turns, randomized=False, seed=args.seed)
    add_expert_traces(records, seen, episodes=args.episodes, max_turns=args.max_turns, randomized=True, seed=args.seed)
    add_recovery_traces(records, seen, episodes=max(90, args.episodes // 3), seed=args.seed)
    add_schema_coverage(records, seen, episodes=36, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(json.dumps(summarize(records, str(output)), indent=2))


def add_expert_traces(
    records: list[dict[str, Any]],
    seen: set[str],
    *,
    episodes: int,
    max_turns: int,
    randomized: bool,
    seed: str,
) -> None:
    with_training_randomization(randomized, seed)
    env = MolForgeEnvironment()
    source = "expert_randomized" if randomized else "expert_canonical"

    for _ in range(episodes):
        observation = env.reset()
        for _ in range(max_turns):
            if observation.done:
                break
            action = heuristic_team_action(observation)
            add_record(records, seen, observation, action, source=source)
            observation = env.step(action)


def add_recovery_traces(records: list[dict[str, Any]], seen: set[str], *, episodes: int, seed: str) -> None:
    with_training_randomization(True, f"{seed}-recovery")
    env = MolForgeEnvironment()

    for episode_index in range(episodes):
        observation = env.reset()

        # Move some episodes to a useful intermediate state before injecting a bad decision.
        for _ in range(episode_index % 3):
            if observation.done:
                break
            observation = env.step(heuristic_team_action(observation))
        if observation.done:
            continue

        for bad_action in bad_actions_for(observation):
            trial = clone_env_at_observation(env, episode_index)
            trial_obs = advance_like_source(trial, episode_index % 3)
            if trial_obs.done:
                continue
            veto_obs = trial.step(attach_team_messages(trial_obs, attach_reasoning_fields(trial_obs, bad_action)))
            if veto_obs.done:
                continue
            if veto_obs.governance.status != "policy_veto":
                continue
            recovery = heuristic_team_action(veto_obs)
            add_record(records, seen, veto_obs, recovery, source="recovery_after_veto")


def add_schema_coverage(records: list[dict[str, Any]], seen: set[str], *, episodes: int, seed: str) -> None:
    with_training_randomization(True, f"{seed}-coverage")
    env = MolForgeEnvironment()
    observations: list[MolForgeObservation] = []
    for _ in range(episodes):
        observation = env.reset()
        observations.append(observation)
        for _ in range(2):
            if observation.done:
                break
            observation = env.step(heuristic_team_action(observation))
            observations.append(observation)

    defer_examples = 0
    for observation in observations:
        current = {slot.slot: slot.fragment for slot in observation.molecule_slots}
        safe_edits = [
            ("solvent_tail", "morpholine", "Use morpholine to reduce safety risk."),
            ("back_pocket", "cyano", "Use cyano to preserve potency with lower lipophilic risk."),
            ("warhead", "reversible_cyanoacrylamide", "Use a softer warhead to reduce reactivity."),
            ("hinge", "azaindole", "Use azaindole when potency needs recovery."),
        ]
        for slot, fragment, rationale in safe_edits:
            if current.get(slot) == fragment:
                continue
            add_record(
                records,
                seen,
                observation,
                MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="substitute",
                    slot=slot,  # type: ignore[arg-type]
                    fragment=fragment,
                    rationale=rationale,
                ),
                source="schema_safe_edit",
            )

        if observation.step_index > 0:
            add_record(
                records,
                seen,
                observation,
                MolForgeAction(
                    action_type="edit",
                    acting_role="lead_chemist",
                    edit_type="remove",
                    slot="back_pocket",
                    rationale="Remove the back-pocket group to simplify risk before reassay.",
                ),
                source="schema_remove",
            )

        for tool_name in useful_tool_subset(observation):
            add_record(
                records,
                seen,
                observation,
                MolForgeAction(
                    action_type="run_assay",
                    acting_role="assay_planner",
                    tool_name=tool_name,  # type: ignore[arg-type]
                    rationale=f"Run {tool_name} to close a visible evidence gap.",
                ),
                source="schema_tool_coverage",
            )

        if (
            defer_examples < 36
            and observation.step_index >= 1
            and observation.scenario_id != "level_2_hard"
        ):
            add_record(
                records,
                seen,
                observation,
                MolForgeAction(
                    action_type="defer",
                    acting_role="lead_chemist",
                    rationale="Defer because no safe evidence-backed action remains in the current budget window.",
                ),
                source="schema_defer",
            )
            defer_examples += 1


def useful_tool_subset(observation: MolForgeObservation) -> list[str]:
    gaps = set()
    for constraint in observation.constraint_status:
        if constraint.evidence_status == "unknown":
            if constraint.name == "toxicity_max":
                gaps.add("toxicity")
            else:
                gaps.add(constraint.name.split("_")[0])
    tools: list[str] = []
    if "potency" in gaps and observation.remaining_budget >= DEFAULT_TOOL_COSTS["dock_target"]:
        tools.extend(["evaluate_properties", "dock_target"])
    if "toxicity" in gaps and observation.remaining_budget >= DEFAULT_TOOL_COSTS["assay_toxicity"]:
        tools.append("assay_toxicity")
    if "synth" in gaps and observation.remaining_budget >= DEFAULT_TOOL_COSTS["estimate_synthesizability"]:
        tools.append("estimate_synthesizability")
    if observation.remaining_budget >= DEFAULT_TOOL_COSTS["evaluate_novelty"]:
        tools.append("evaluate_novelty")
    if observation.remaining_budget >= DEFAULT_TOOL_COSTS["search_literature"]:
        tools.append("search_literature")
    if observation.scenario_id == "level_2_hard" and observation.remaining_budget >= DEFAULT_TOOL_COSTS["run_md_simulation"]:
        tools.append("run_md_simulation")
    return tools


def bad_actions_for(observation: MolForgeObservation) -> Iterable[MolForgeAction]:
    current = {slot.slot: slot.fragment for slot in observation.molecule_slots}
    candidates = [
        ("solvent_tail", "dimethylamino", "This would add a safety liability and should be recovered from."),
        ("back_pocket", "trifluoromethyl", "This would over-shoot lipophilic risk and should be recovered from."),
        ("hinge", "quinazoline", "This can create route pressure and should be recovered from."),
    ]
    for slot, fragment, rationale in candidates:
        if current.get(slot) == fragment:
            continue
        yield MolForgeAction(
            action_type="edit",
            acting_role="lead_chemist",
            edit_type="substitute",
            slot=slot,  # type: ignore[arg-type]
            fragment=fragment,
            rationale=rationale,
        )


def clone_env_at_observation(source_env: MolForgeEnvironment, episode_index: int) -> MolForgeEnvironment:
    del source_env
    env = MolForgeEnvironment()
    for _ in range(episode_index + 1):
        observation = env.reset()
    return env


def advance_like_source(env: MolForgeEnvironment, steps: int) -> MolForgeObservation:
    observation = env._build_observation(reward=0.0, done=False, reward_components=[])  # noqa: SLF001
    for _ in range(steps):
        if observation.done:
            return observation
        observation = env.step(heuristic_team_action(observation))
    return observation


def with_training_randomization(enabled: bool, seed: str) -> None:
    if enabled:
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"
    else:
        os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
    os.environ["MOLFORGE_RANDOM_SEED"] = seed


def add_record(
    records: list[dict[str, Any]],
    seen: set[str],
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> None:
    action = attach_reasoning_fields(observation, action)
    record = make_record(observation, action, source=source)
    key = json.dumps(
        {"user": record["messages"][1]["content"], "assistant": record["messages"][2]["content"]},
        sort_keys=True,
    )
    if key in seen:
        return
    validate_target(record["messages"][2]["content"])
    records.append(record)
    seen.add(key)


def make_record(observation: MolForgeObservation, action: MolForgeAction, *, source: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": COMPACT_ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(compact_action_payload(observation), separators=(",", ":"))},
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


def compact_action_payload(observation: MolForgeObservation) -> dict[str, Any]:
    lead_view = next(
        (role.observation for role in observation.role_observations if role.role == "lead_chemist"),
        {},
    )
    assay_view = next(
        (role.observation for role in observation.role_observations if role.role == "assay_planner"),
        {},
    )
    return {
        "valid_action_types": ["edit", "run_assay", "submit", "restart", "defer"],
        "scenario_id": observation.scenario_id,
        "difficulty": observation.difficulty,
        "task_brief": observation.task_brief,
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
        "tool_costs": assay_view.get("tool_costs", {}),
        "evidence_gaps": assay_view.get("evidence_gaps", []),
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
    if data["action_type"] not in {"edit", "run_assay", "submit", "restart", "defer"}:
        raise ValueError(f"invalid action_type: {data['action_type']}")
    if data["action_type"] == "proposal":
        raise ValueError("proposal is not a compact action type")
    if data["edit_type"] == "replace":
        raise ValueError("replace must never be used; use substitute")
    if "messages" in data:
        raise ValueError("compact target must not contain messages")
    if not isinstance(data["evidence"], list):
        raise ValueError("evidence must be a list")
    if set(data["expected_effects"]) != {"potency", "toxicity", "synth", "novelty", "budget"}:
        raise ValueError("expected_effects must have exactly five keys")
    MolForgeAction(**data)


def summarize(records: list[dict[str, Any]], output: str) -> dict[str, Any]:
    actions: dict[str, int] = {}
    sources: dict[str, int] = {}
    scenarios: dict[str, int] = {}
    users = set()
    assistants = set()
    for record in records:
        metadata = record["metadata"]
        actions[metadata["action_type"]] = actions.get(metadata["action_type"], 0) + 1
        sources[metadata["source"]] = sources.get(metadata["source"], 0) + 1
        scenarios[metadata["scenario_id"]] = scenarios.get(metadata["scenario_id"], 0) + 1
        users.add(record["messages"][1]["content"])
        assistants.add(record["messages"][2]["content"])
    return {
        "output": output,
        "records": len(records),
        "unique_user_prompts": len(users),
        "unique_assistant_targets": len(assistants),
        "action_types": actions,
        "sources": sources,
        "scenario_ids": scenarios,
    }


if __name__ == "__main__":
    main()
