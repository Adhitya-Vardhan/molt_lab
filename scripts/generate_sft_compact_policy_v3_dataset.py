"""Generate policy-focused compact MolForge SFT data.

This dataset is for improving decision quality after the compact schema is
mostly learned. It uses the successful heuristic team policy on randomized
training variants and avoids broad coverage edits that taught the model to edit
unsafe fragments at the first step.
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

from inference_common import heuristic_team_action  # noqa: E402
from server.molforge_environment import MolForgeEnvironment  # noqa: E402
from scripts.generate_sft_compact_actions_v2_dataset import (  # noqa: E402
    make_record,
    validate_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate policy-focused compact MolForge SFT JSONL.")
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", default="policy-v3")
    parser.add_argument("--output", default="issue/molforge_sft_compact_policy_v3.jsonl")
    args = parser.parse_args()

    os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"
    os.environ["MOLFORGE_RANDOM_SEED"] = args.seed

    env = MolForgeEnvironment()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _ in range(args.episodes):
        observation = env.reset()
        for _ in range(args.max_turns):
            if observation.done:
                break
            action = heuristic_team_action(observation)
            record = make_record(observation, action, source="policy_v3")
            key = json.dumps(
                {
                    "user": record["messages"][1]["content"],
                    "assistant": record["messages"][2]["content"],
                },
                sort_keys=True,
            )
            if key not in seen:
                validate_target(record["messages"][2]["content"])
                records.append(record)
                seen.add(key)
            observation = env.step(action)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(json.dumps(summarize(records, str(output)), indent=2))


def summarize(records: list[dict[str, Any]], output: str) -> dict[str, Any]:
    actions: dict[str, int] = {}
    scenarios: dict[str, int] = {}
    users = set()
    assistants = set()
    for record in records:
        metadata = record["metadata"]
        actions[metadata["action_type"]] = actions.get(metadata["action_type"], 0) + 1
        scenarios[metadata["scenario_id"]] = scenarios.get(metadata["scenario_id"], 0) + 1
        users.add(record["messages"][1]["content"])
        assistants.add(record["messages"][2]["content"])
    return {
        "output": output,
        "records": len(records),
        "unique_user_prompts": len(users),
        "unique_assistant_targets": len(assistants),
        "action_types": actions,
        "scenario_ids": scenarios,
    }


if __name__ == "__main__":
    main()
