"""Generate a MolForge SFT JSONL dataset with rare-action coverage.

Most records come from the deterministic team policy so the examples are
grounded in real environment trajectories. A smaller coverage slice is added
for rare but valid schema variants such as defer, each assay tool, and edit
subtypes so SFT teaches the model the whole action surface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_common import (  # noqa: E402
    SYSTEM_PROMPT,
    MolForgeAction,
    MolForgeObservation,
    attach_reasoning_fields,
    attach_team_messages,
    build_model_payload,
    heuristic_team_action,
)
from scenarios import DEFAULT_TOOL_COSTS  # noqa: E402
from server.molforge_environment import MolForgeEnvironment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MolForge all-action SFT JSONL.")
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--output", default="data/molforge_sft_all_actions.jsonl")
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
    records = []

    for _ in range(args.episodes):
        observation = env.reset()
        for _ in range(args.max_turns):
            if observation.done:
                break
            action = heuristic_team_action(observation)
            records.append(make_record(observation, action, source="policy_trace"))
            observation = env.step(action)

    for observation, action in curated_coverage_examples():
        action = attach_reasoning_fields(observation, action)
        action = attach_team_messages(observation, action)
        records.append(make_record(observation, action, source="coverage_example"))

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "records": len(records),
                "coverage_records": sum(
                    1 for record in records if record["metadata"]["source"] == "coverage_example"
                ),
            },
            indent=2,
        )
    )


def curated_coverage_examples() -> Iterable[tuple[MolForgeObservation, MolForgeAction]]:
    env = MolForgeEnvironment()
    observations = [env.reset(), env.reset(), env.reset()]

    for observation in observations:
        yield observation, MolForgeAction(
            action_type="defer",
            acting_role="lead_chemist",
            rationale="Hold this turn because the team needs a cleaner evidence-backed move.",
        )

    easy, medium, hard = observations

    yield easy, MolForgeAction(
        action_type="edit",
        acting_role="lead_chemist",
        edit_type="add_fragment",
        slot="back_pocket",
        fragment="cyano",
        rationale="Add a compact cyano handle to improve potency without large lipophilic risk.",
    )
    yield medium, MolForgeAction(
        action_type="edit",
        acting_role="lead_chemist",
        edit_type="remove",
        slot="back_pocket",
        rationale="Remove the risky back-pocket group and return to a simpler default handle.",
    )
    yield hard, MolForgeAction(
        action_type="edit",
        acting_role="lead_chemist",
        edit_type="undo_last_edit",
        slot="solvent_tail",
        rationale="Undo the last tail change when the visible evidence suggests it raised risk.",
    )

    for observation in observations:
        for tool_name in DEFAULT_TOOL_COSTS:
            yield observation, MolForgeAction(
                action_type="run_assay",
                acting_role="assay_planner",
                tool_name=tool_name,
                rationale=f"Run {tool_name} to close a visible evidence gap before committing.",
            )

    yield hard, MolForgeAction(
        action_type="restart",
        acting_role="lead_chemist",
        rationale="Restart early because the hard scenario starts in a trap series.",
    )
    yield easy, MolForgeAction(
        action_type="submit",
        acting_role="lead_chemist",
        rationale="Submit only when visible evidence is sufficient and budget should be preserved.",
    )


def make_record(
    observation: MolForgeObservation,
    action: MolForgeAction,
    *,
    source: str,
) -> dict[str, object]:
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
                "content": json.dumps(
                    action.model_dump(exclude_none=True),
                    separators=(",", ":"),
                ),
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


if __name__ == "__main__":
    main()
