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


def load_completion_diagnostics(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL completion diagnostics emitted during GRPO training."""

    log_path = Path(path)
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def generate_training_artifacts(
    *,
    log_history: list[dict[str, Any]],
    completion_log_path: Path | str,
    output_dir: Path | str,
) -> dict[str, str]:
    """Create post-training plots from TRL logs and MolForge completion diagnostics."""

    import matplotlib.pyplot as plt
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    history_df = pd.DataFrame(log_history)
    diagnostics = load_completion_diagnostics(completion_log_path)
    diagnostics_df = pd.DataFrame(diagnostics)

    if not history_df.empty:
        history_df["step"] = pd.to_numeric(history_df.get("step"), errors="coerce")

    reward_col = _first_present_column(history_df, ["reward", "rewards/mean", "train_reward"])
    if reward_col is not None:
        reward_df = history_df.dropna(subset=["step", reward_col]).copy()
        if not reward_df.empty:
            reward_df["reward_ma"] = reward_df[reward_col].rolling(window=min(20, len(reward_df)), min_periods=1).mean()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(reward_df["step"], reward_df[reward_col], alpha=0.35, label="raw reward")
            ax.plot(reward_df["step"], reward_df["reward_ma"], linewidth=2, label="moving average")
            ax.set_title("GRPO Reward Curve")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.legend()
            fig.tight_layout()
            path = output / "reward_curve.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved["reward_curve"] = str(path)

    loss_col = _first_present_column(history_df, ["loss", "train/loss"])
    if loss_col is not None:
        loss_df = history_df.dropna(subset=["step", loss_col]).copy()
        if not loss_df.empty:
            loss_df["loss_ma"] = loss_df[loss_col].rolling(window=min(20, len(loss_df)), min_periods=1).mean()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(loss_df["step"], loss_df[loss_col], alpha=0.35, label="raw loss")
            ax.plot(loss_df["step"], loss_df["loss_ma"], linewidth=2, label="moving average")
            ax.set_title("GRPO Loss Curve")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
            path = output / "loss_curve.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved["loss_curve"] = str(path)

    if not diagnostics_df.empty and "action_type" in diagnostics_df:
        action_counts = diagnostics_df["action_type"].fillna("unknown").value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        action_counts.plot(kind="bar", ax=ax)
        ax.set_title("Action Distribution")
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = output / "action_distribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved["action_distribution"] = str(path)

    if not diagnostics_df.empty and "reward" in diagnostics_df:
        diagnostics_df["index"] = range(1, len(diagnostics_df) + 1)
        diagnostics_df["reward_ma"] = diagnostics_df["reward"].rolling(
            window=min(25, len(diagnostics_df)),
            min_periods=1,
        ).mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(diagnostics_df["index"], diagnostics_df["reward"], alpha=0.30, label="completion reward")
        ax.plot(diagnostics_df["index"], diagnostics_df["reward_ma"], linewidth=2, label="moving average")
        ax.set_title("Completion Reward Trend")
        ax.set_xlabel("Completion")
        ax.set_ylabel("Reward")
        ax.legend()
        fig.tight_layout()
        path = output / "completion_reward_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved["completion_reward_curve"] = str(path)

    final_scores = _score_series(diagnostics, "final_score")
    if final_scores:
        score_df = pd.DataFrame({"index": range(1, len(final_scores) + 1), "final_score": final_scores})
        score_df["final_score_ma"] = score_df["final_score"].rolling(
            window=min(25, len(score_df)),
            min_periods=1,
        ).mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(score_df["index"], score_df["final_score"], alpha=0.30, label="completion final_score")
        ax.plot(score_df["index"], score_df["final_score_ma"], linewidth=2, label="moving average")
        ax.set_title("Final Score Trend")
        ax.set_xlabel("Completion")
        ax.set_ylabel("final_score")
        ax.legend()
        fig.tight_layout()
        path = output / "final_score_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved["final_score_curve"] = str(path)

    return saved


def _first_present_column(frame: Any, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame:
            return candidate
    return None


def _score_series(records: list[dict[str, Any]], score_name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        score_block = record.get("scores", {})
        value = score_block.get(score_name)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values
