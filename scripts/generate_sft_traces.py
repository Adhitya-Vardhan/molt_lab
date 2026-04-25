"""Generate small MolForge SFT datasets from the heuristic team policy.

The output is JSONL with chat-style records:
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
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

from inference_common import SYSTEM_PROMPT, build_model_payload, heuristic_team_action

try:
    from molforge.server.molforge_environment import MolForgeEnvironment
except ImportError:
    from server.molforge_environment import MolForgeEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MolForge SFT traces.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--output", default="data/molforge_sft_traces.jsonl")
    parser.add_argument(
        "--randomized",
        action="store_true",
        help="Enable MolForge training randomization while collecting traces.",
    )
    args = parser.parse_args()

    if args.randomized:
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = MolForgeEnvironment()
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for _ in range(args.episodes):
            observation = env.reset()
            for _ in range(args.max_turns):
                if observation.done:
                    break
                action = heuristic_team_action(observation)
                record = {
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
                        "scenario_id": observation.scenario_id,
                        "difficulty": observation.difficulty,
                        "step_index": observation.step_index,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                written += 1
                observation = env.step(action)

    summary: dict[str, Any] = {"output": str(output_path), "records": written}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
