"""Shared RL notebook helpers for MolForge GRPO training."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    from datasets import Dataset
except Exception:  # pragma: no cover - optional local dependency
    Dataset = None  # type: ignore[assignment]

from inference_common import (
    MolForgeAction,
    attach_reasoning_fields,
    attach_team_messages,
    extract_json,
    heuristic_team_action,
)
from scripts.generate_sft_compact_policy_v4_dataset import (
    COMPACT_ACTION_SYSTEM_PROMPT,
    compact_action_payload,
)
from server.molforge_environment import MolForgeEnvironment

COMPLETION_LOG = Path(os.getenv("MOLFORGE_COMPLETION_LOG", "completion_diagnostics.jsonl"))


def replay_to_state(record: dict[str, Any]) -> MolForgeEnvironment:
    """Replay a stored prompt record to the exact environment state it came from."""

    if record.get("randomized"):
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"
    else:
        os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
    os.environ["MOLFORGE_RANDOM_SEED"] = str(record.get("random_seed", "rl"))

    env = MolForgeEnvironment()
    observation = env.reset()
    for action_payload in record.get("pre_actions", []):
        action = MolForgeAction(**action_payload)
        # Backward-compatible repair for old cached records that stored partial actions.
        if not action.messages or not action.evidence or not action.rationale:
            action = attach_team_messages(observation, attach_reasoning_fields(observation, action))
        observation = env.step(action)
    return env


def evaluate_completion(
    prompt_str: str,
    completion_str: str,
    record: dict[str, Any],
) -> Tuple[float, dict[str, Any]]:
    """Score one generated action against a replayed MolForge state."""

    del prompt_str
    try:
        action_dict = extract_json(completion_str)
        action = MolForgeAction(**action_dict)
    except Exception:
        return -1.5, {"valid_json": False, "action_type": "invalid"}

    env = replay_to_state(record)
    observation = env._build_observation(reward=0.0, done=False, reward_components=[])  # noqa: SLF001
    previous_scores = env._grade_all()  # noqa: SLF001

    action = attach_team_messages(observation, attach_reasoning_fields(observation, action))
    next_observation = env.step(action)
    next_scores = env._grade_all()  # noqa: SLF001

    component_values = {
        component.name: component.value
        for component in next_observation.reward_breakdown
    }

    candidate_delta = next_scores["candidate_score"] - previous_scores["candidate_score"]
    constraint_delta = next_scores["constraint_margin_score"] - previous_scores["constraint_margin_score"]
    evidence_delta = next_scores["evidence_score"] - previous_scores["evidence_score"]
    chemistry_delta = next_scores["chemical_quality_score"] - previous_scores["chemical_quality_score"]

    reward = (
        1.8 * candidate_delta
        + 1.2 * constraint_delta
        + 0.45 * evidence_delta
        + 0.35 * chemistry_delta
    )

    reward += component_values.get("invalid_action", 0.0)
    reward += component_values.get("budget_exhausted", 0.0)
    reward += component_values.get("step_limit", 0.0)
    reward += component_values.get("policy_veto", 0.0)
    reward += component_values.get("loop_penalty", 0.0)
    reward += component_values.get("chemical_validity", 0.0)

    if action.action_type == "edit":
        reward += 0.08
        if candidate_delta <= 0.0 and constraint_delta <= 0.0:
            reward -= 0.18
    elif action.action_type == "run_assay":
        reward -= 0.12
        if evidence_delta <= 0.0:
            reward -= 0.18
        else:
            reward += 0.06 * min(1.0, evidence_delta * 3.0)
    elif action.action_type == "restart":
        reward -= 0.10
    elif action.action_type == "defer":
        reward -= 0.25
    elif action.action_type == "submit":
        submission_score = next_scores.get("submission_score", 0.0)
        if submission_score > 0.0:
            reward += 1.5 + 4.0 * submission_score
        else:
            reward -= 0.75

    if not next_observation.done:
        reward -= 0.08

    reward = round(reward, 4)
    return reward, {
        "valid_json": True,
        "action_type": action.action_type,
        "reward": reward,
        "done": next_observation.done,
        "scores": next_scores,
        "reward_components": component_values,
        "raw_completion": completion_str,
        "timestamp": time.time(),
    }


def molforge_reward_func(prompts, completions, **kwargs) -> list[float]:
    """TRL reward function wrapper for GRPO."""

    del prompts
    rewards = []
    records = kwargs.get("record", [])
    COMPLETION_LOG.parent.mkdir(parents=True, exist_ok=True)

    for i in range(len(completions)):
        record = records[i] if i < len(records) else {}
        reward, diagnostics = evaluate_completion("", completions[i][0]["content"], record)
        rewards.append(reward)
        with COMPLETION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostics, ensure_ascii=True) + "\n")
    return rewards


def build_dynamic_prompts(
    episodes: int = 50,
    max_turns: int = 5,
    *,
    randomized: bool = True,
    seed: str = "dynamic-rl",
) -> Any:
    """Generate compact RL prompts by rolling out the heuristic policy."""

    if randomized:
        os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"
    else:
        os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
    os.environ["MOLFORGE_RANDOM_SEED"] = seed

    records = []
    env = MolForgeEnvironment()

    for _ in range(episodes):
        observation = env.reset()
        pre_actions: list[dict[str, Any]] = []

        for _ in range(max_turns):
            if observation.done:
                break

            prompt_payload = compact_action_payload(observation)
            records.append(
                {
                    "prompt": [
                        {"role": "system", "content": COMPACT_ACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=True)},
                    ],
                    "record": {
                        "scenario_id": observation.scenario_id,
                        "difficulty": observation.difficulty,
                        "step_index": observation.step_index,
                        "pre_actions": list(pre_actions),
                        "randomized": randomized,
                        "random_seed": seed,
                    },
                }
            )

            action = heuristic_team_action(observation)
            observation = env.step(action)
            pre_actions.append(action.model_dump(exclude_none=False))

    random.shuffle(records)
    if Dataset is None:
        return records
    return Dataset.from_list(records)
