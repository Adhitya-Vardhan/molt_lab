# -*- coding: utf-8 -*-
"""MolForge GRPO RL training script for Colab/Kaggle.

This script continues from the SFT LoRA adapter and trains against the real
MolForge environment reward. It writes rich debug logs, metrics CSV/JSON, plots,
and adapter checkpoints so reward curves can be shown in the hackathon demo.

Recommended Colab setup:

    !pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install -U "trl>=0.21.0" peft accelerate bitsandbytes datasets matplotlib pandas

    import os
    os.environ["SFT_ADAPTER_PATH"] = "/content/drive/MyDrive/.../qwen3_5_2b_lora_adapters_compact_v4"
    os.environ["DRIVE_OUTPUT_DIR"] = "/content/drive/MyDrive/MolForge_RL_Runs"
    os.environ["RL_MAX_STEPS"] = "80"
    os.environ["NUM_GENERATIONS"] = "2"
    !python issue/qwen3_5_2b_unsloth_grpo_rl.py
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_URL = os.getenv("MOLFORGE_REPO_URL", "https://github.com/Adhitya-Vardhan/molt_lab.git")
DEFAULT_PROJECT_ROOT = "/content/molt_lab" if Path("/content").exists() else "/kaggle/working/molt_lab"
PROJECT_ROOT = Path(os.getenv("MOLFORGE_PROJECT_ROOT", DEFAULT_PROJECT_ROOT))


def ensure_project_available() -> Path:
    """Clone the repo on Kaggle if the project files are not already present."""

    if Path("server/molforge_environment.py").exists():
        root = Path.cwd()
    elif (PROJECT_ROOT / "server/molforge_environment.py").exists():
        root = PROJECT_ROOT
    else:
        PROJECT_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(PROJECT_ROOT)], check=True)
        root = PROJECT_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


ROOT = ensure_project_available()

os.environ.setdefault("MOLFORGE_REWARD_MODE", "curriculum")
os.environ.setdefault("MOLFORGE_TRAINING_RANDOMIZATION", "1")
os.environ.pop("MOLFORGE_DEBUG_STATE", None)

if Path("/content").exists() and not Path("/kaggle/working").exists():
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Skipping Google Drive mount: {exc}")

from datasets import Dataset  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
except Exception:  # pragma: no cover - script can fall back to Transformers.
    FastLanguageModel = None

    def is_bfloat16_supported() -> bool:
        return False

from inference_common import (  # noqa: E402
    MolForgeAction,
    attach_reasoning_fields,
    attach_team_messages,
    extract_json,
    heuristic_team_action,
)
from scripts.generate_sft_compact_policy_v4_dataset import (  # noqa: E402
    COMPACT_ACTION_SYSTEM_PROMPT,
    compact_action_payload,
)
from server.molforge_environment import MolForgeEnvironment  # noqa: E402


BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/Qwen3.5-2B")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1536"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "384"))
RL_MAX_STEPS = int(os.getenv("RL_MAX_STEPS", "80"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
PROMPT_EPISODES = int(os.getenv("RL_PROMPT_EPISODES", "96"))
PROMPT_MAX_TURNS = int(os.getenv("RL_PROMPT_MAX_TURNS", "8"))
EVAL_MAX_TURNS = int(os.getenv("EVAL_MAX_TURNS", "10"))
RUN_EVAL = os.getenv("RUN_EVAL", "1").lower() in {"1", "true", "yes"}
LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "2e-6"))
PER_DEVICE_BATCH = int(os.getenv("RL_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("RL_GRAD_ACCUM", "4"))
SAVE_STEPS = int(os.getenv("RL_SAVE_STEPS", "25"))
LOGGING_STEPS = int(os.getenv("RL_LOGGING_STEPS", "1"))
SEED = int(os.getenv("RL_SEED", "3407"))
USE_UNSLOTH = os.getenv("USE_UNSLOTH", "true").lower() in {"1", "true", "yes"}
RUN_NAME = os.getenv("RUN_NAME", time.strftime("molforge_grpo_%Y%m%d_%H%M%S"))
DEFAULT_OUTPUT_ROOT = "/content/molforge_rl_runs" if Path("/content").exists() else "/kaggle/working/molforge_rl_runs"
OUTPUT_ROOT = Path(os.getenv("RL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
ADAPTER_SAVE_DIR = OUTPUT_DIR / "adapters"
DRIVE_OUTPUT_DIR = os.getenv("DRIVE_OUTPUT_DIR", "")


def find_sft_adapter_path() -> str:
    explicit = os.getenv("SFT_ADAPTER_PATH")
    if explicit:
        return explicit
    candidates = [
        Path("/content/drive/MyDrive/Qwen_3.5_finetune/qwen3_5_2b_lora_adapters_compact_v4"),
        Path("/kaggle/working/qwen3_5_2b_lora_adapters_policy_v4"),
        Path("/content/drive/MyDrive/Qwen_3.5_finetune/qwen3_5_2b_lora_adapters_policy_v4"),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.rglob("qwen3_5_2b_lora_adapters_compact_v4")))
        candidates.extend(sorted(kaggle_input.rglob("qwen3_5_2b_lora_adapters_policy_v4")))
        candidates.extend(sorted(kaggle_input.rglob("adapter_config.json")))
    for candidate in candidates:
        if candidate.name == "adapter_config.json":
            candidate = candidate.parent
        if (candidate / "adapter_config.json").exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find SFT adapter. Set SFT_ADAPTER_PATH to the LoRA adapter directory."
    )


SFT_ADAPTER_PATH = find_sft_adapter_path()

for path in [OUTPUT_DIR, LOG_DIR, PLOT_DIR, ADAPTER_SAVE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

COMPLETION_LOG = LOG_DIR / "completion_rewards.jsonl"
TRAINER_LOG = LOG_DIR / "trainer_log_history.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "rl_summary.json"
METRICS_CSV = OUTPUT_DIR / "completion_metrics.csv"
RUN_MANIFEST_JSON = OUTPUT_DIR / "run_manifest.json"
EVAL_BEFORE_JSON = OUTPUT_DIR / "eval_before_training.json"
EVAL_AFTER_JSON = OUTPUT_DIR / "eval_after_training.json"


def as_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def action_to_compact_dict(action: MolForgeAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type,
        "acting_role": action.acting_role,
        "edit_type": action.edit_type,
        "slot": action.slot,
        "fragment": action.fragment,
        "tool_name": action.tool_name,
        "rationale": action.rationale,
        "evidence": list(action.evidence[:5]),
        "expected_effects": {
            "potency": action.expected_effects.get("potency", "unknown"),
            "toxicity": action.expected_effects.get("toxicity", "unknown"),
            "synth": action.expected_effects.get("synth", "unknown"),
            "novelty": action.expected_effects.get("novelty", "unknown"),
            "budget": action.expected_effects.get("budget", "neutral"),
        },
    }


def replay_to_state(record: dict[str, Any]) -> MolForgeEnvironment:
    previous_randomization = os.environ.get("MOLFORGE_TRAINING_RANDOMIZATION")
    previous_seed = os.environ.get("MOLFORGE_RANDOM_SEED")
    try:
        if record["randomized"]:
            os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"
        else:
            os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
        os.environ["MOLFORGE_RANDOM_SEED"] = record["random_seed"]
        env = MolForgeEnvironment()
        observation = None
        for _ in range(record["reset_count"] + 1):
            observation = env.reset()
        assert observation is not None
        for action_payload in record["pre_actions"]:
            action = MolForgeAction(**action_payload)
            observation = env.step(attach_team_messages(observation, attach_reasoning_fields(observation, action)))
            if observation.done:
                break
        return env
    finally:
        if previous_randomization is None:
            os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
        else:
            os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = previous_randomization
        if previous_seed is None:
            os.environ.pop("MOLFORGE_RANDOM_SEED", None)
        else:
            os.environ["MOLFORGE_RANDOM_SEED"] = previous_seed


def build_prompt_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    modes = [
        ("canonical", False, "rl-canonical", max(9, PROMPT_EPISODES // 5)),
        ("randomized", True, "rl-randomized", PROMPT_EPISODES),
    ]
    for source, randomized, seed, episodes in modes:
        if randomized:
            os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = "1"
        else:
            os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
        os.environ["MOLFORGE_RANDOM_SEED"] = seed
        env = MolForgeEnvironment()
        for reset_count in range(episodes):
            observation = env.reset()
            pre_actions: list[dict[str, Any]] = []
            for _ in range(PROMPT_MAX_TURNS):
                if observation.done:
                    break
                prompt_payload = compact_action_payload(observation)
                records.append(
                    {
                        "prompt": [
                            {"role": "system", "content": COMPACT_ACTION_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(prompt_payload, separators=(",", ":"))},
                        ],
                        "scenario_id": observation.scenario_id,
                        "difficulty": observation.difficulty,
                        "step_index": observation.step_index,
                        "source": source,
                        "randomized": randomized,
                        "random_seed": seed,
                        "reset_count": reset_count,
                        "pre_actions": list(pre_actions),
                    }
                )
                expert_action = heuristic_team_action(observation)
                pre_actions.append(action_to_compact_dict(expert_action))
                observation = env.step(expert_action)
    random.Random(SEED).shuffle(records)
    limit = int(os.getenv("RL_PROMPT_LIMIT", "0"))
    if limit > 0:
        records = records[:limit]
    return records


def evaluate_completion(record: dict[str, Any], completion_text: str) -> tuple[float, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "scenario_id": record["scenario_id"],
        "difficulty": record["difficulty"],
        "step_index": record["step_index"],
        "source": record["source"],
        "raw_completion": completion_text[:2000],
    }
    try:
        parsed = extract_json(completion_text)
        diagnostics["parsed_action_type"] = parsed.get("action_type")
        action = MolForgeAction(**parsed)
    except Exception as exc:
        diagnostics.update(
            {
                "valid_json": False,
                "error": f"{exc.__class__.__name__}: {exc}",
                "reward": -1.2,
            }
        )
        return -1.2, diagnostics

    diagnostics["valid_json"] = True
    env = replay_to_state(record)
    observation = env._build_observation(reward=0.0, done=False, reward_components=[])  # noqa: SLF001
    action = attach_team_messages(observation, attach_reasoning_fields(observation, action))
    next_observation = env.step(action)
    grader_scores = next_observation.metadata.get("terminal_grader_scores", {})
    components = {component.name: component.value for component in next_observation.reward_breakdown}
    reward = float(next_observation.reward)

    diagnostics.update(
        {
            "action_type": action.action_type,
            "acting_role": action.acting_role,
            "tool_name": action.tool_name,
            "slot": action.slot,
            "fragment": action.fragment,
            "reward": reward,
            "done": next_observation.done,
            "governance_status": next_observation.governance.status,
            "remaining_budget": next_observation.remaining_budget,
            "budget_used": next_observation.budget_used,
            "last_transition_summary": next_observation.last_transition_summary,
            "reward_components": components,
            "final_score": grader_scores.get("final_score"),
            "submission_score": grader_scores.get("submission_score"),
            "progress_score": grader_scores.get("progress_score"),
            "candidate_score": grader_scores.get("candidate_score"),
            "evidence_score": grader_scores.get("evidence_score"),
            "budget_score": grader_scores.get("budget_score"),
            "coordination_score": grader_scores.get("coordination_score"),
        }
    )
    return reward, diagnostics


def molforge_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards: list[float] = []
    batch_rows: list[dict[str, Any]] = []
    record_count = len(completions)
    for index in range(record_count):
        record = {
            "scenario_id": kwargs["scenario_id"][index],
            "difficulty": kwargs["difficulty"][index],
            "step_index": kwargs["step_index"][index],
            "source": kwargs["source"][index],
            "randomized": kwargs["randomized"][index],
            "random_seed": kwargs["random_seed"][index],
            "reset_count": kwargs["reset_count"][index],
            "pre_actions": kwargs["pre_actions"][index],
        }
        completion_text = as_completion_text(completions[index])
        reward, diagnostics = evaluate_completion(record, completion_text)
        rewards.append(reward)
        diagnostics["timestamp"] = time.time()
        batch_rows.append(diagnostics)

    with COMPLETION_LOG.open("a", encoding="utf-8") as handle:
        for row in batch_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return rewards


class JsonMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if logs:
            row = {"step": state.global_step, "time": time.time(), **logs}
            with TRAINER_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def on_save(self, args, state, control, **kwargs):
        write_summary_and_plots()


def load_model_and_tokenizer():
    if USE_UNSLOTH and FastLanguageModel is not None:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=SFT_ADAPTER_PATH,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_training(model)
            return model, tokenizer
        except Exception as exc:
            print(f"Unsloth adapter load failed, falling back to Transformers+PEFT: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, is_trainable=True)
    return model, tokenizer


def make_grpo_config() -> GRPOConfig:
    kwargs = {
        "output_dir": str(OUTPUT_DIR / "trainer"),
        "run_name": RUN_NAME,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "max_steps": RL_MAX_STEPS,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "num_generations": NUM_GENERATIONS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "temperature": float(os.getenv("RL_TEMPERATURE", "0.7")),
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "report_to": [],
        "log_completions": os.getenv("RL_LOG_COMPLETIONS", "0") == "1",
        "remove_unused_columns": False,
        "seed": SEED,
    }
    signature = inspect.signature(GRPOConfig.__init__).parameters
    return GRPOConfig(**{key: value for key, value in kwargs.items() if key in signature})


def write_completion_metrics_csv() -> list[dict[str, Any]]:
    if not COMPLETION_LOG.exists():
        return []
    rows = [json.loads(line) for line in COMPLETION_LOG.open(encoding="utf-8")]
    fieldnames = [
        "timestamp",
        "scenario_id",
        "difficulty",
        "step_index",
        "source",
        "valid_json",
        "action_type",
        "parsed_action_type",
        "reward",
        "done",
        "governance_status",
        "remaining_budget",
        "budget_used",
        "final_score",
        "submission_score",
        "progress_score",
        "candidate_score",
        "evidence_score",
        "budget_score",
        "coordination_score",
        "error",
    ]
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary_and_plots() -> None:
    rows = write_completion_metrics_csv()
    trainer_rows = read_jsonl(TRAINER_LOG)
    if not rows:
        return
    rewards = [float(row.get("reward", 0.0)) for row in rows]
    valid_count = sum(1 for row in rows if row.get("valid_json"))
    action_counts = Counter(row.get("action_type") or row.get("parsed_action_type") or "invalid" for row in rows)
    governance_counts = Counter(row.get("governance_status") or "parse_failed" for row in rows)
    scenario_rewards: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scenario_rewards[row.get("scenario_id", "unknown")].append(float(row.get("reward", 0.0)))

    summary = {
        "run_name": RUN_NAME,
        "base_model": BASE_MODEL_NAME,
        "sft_adapter_path": SFT_ADAPTER_PATH,
        "reward_mode": os.getenv("MOLFORGE_REWARD_MODE"),
        "num_logged_completions": len(rows),
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "valid_json_rate": valid_count / max(len(rows), 1),
        "action_counts": dict(action_counts),
        "governance_counts": dict(governance_counts),
        "mean_reward_by_scenario": {
            scenario: sum(values) / max(len(values), 1)
            for scenario, values in scenario_rewards.items()
        },
        "latest_trainer_log": trainer_rows[-1] if trainer_rows else {},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        window = max(4, min(32, len(rewards) // 10 or 4))
        moving = [
            sum(rewards[max(0, i - window + 1) : i + 1]) / len(rewards[max(0, i - window + 1) : i + 1])
            for i in range(len(rewards))
        ]
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.25, label="completion reward")
        plt.plot(moving, label=f"moving avg ({window})")
        plt.xlabel("Logged completion")
        plt.ylabel("MolForge reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "reward_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        labels, counts = zip(*action_counts.items())
        plt.bar(labels, counts)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "action_distribution.png", dpi=160)
        plt.close()

        if trainer_rows:
            plot_trainer_metric(trainer_rows, "loss", PLOT_DIR / "loss_curve.png")
            plot_trainer_metric(trainer_rows, "grad_norm", PLOT_DIR / "grad_norm_curve.png")

        eval_before = read_json_file(EVAL_BEFORE_JSON)
        eval_after = read_json_file(EVAL_AFTER_JSON)
        if eval_before and eval_after:
            plot_eval_comparison(eval_before, eval_after)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def plot_trainer_metric(rows: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    points = [(row.get("step"), row.get(metric)) for row in rows if row.get(metric) is not None]
    if not points:
        return
    steps, values = zip(*points)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.plot(steps, values, marker="o", linewidth=1.5)
    plt.xlabel("Trainer step")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_eval_comparison(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_scores = {row["scenario_id"]: row["final_score"] for row in before.get("episodes", [])}
    after_scores = {row["scenario_id"]: row["final_score"] for row in after.get("episodes", [])}
    labels = sorted(set(before_scores) | set(after_scores))
    if not labels:
        return
    x_positions = list(range(len(labels)))
    width = 0.36
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.bar(
        [value - width / 2 for value in x_positions],
        [before_scores.get(label, 0.0) for label in labels],
        width,
        label="before RL",
    )
    plt.bar(
        [value + width / 2 for value in x_positions],
        [after_scores.get(label, 0.0) for label in labels],
        width,
        label="after RL",
    )
    plt.xticks(x_positions, labels, rotation=20, ha="right")
    plt.ylabel("final_score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "eval_before_after.png", dpi=160)
    plt.close()


def generate_policy_action(model, tokenizer, observation) -> tuple[MolForgeAction | None, str, str]:
    prompt_payload = compact_action_payload(observation)
    messages = [
        {"role": "system", "content": COMPACT_ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt_payload, separators=(",", ":"))},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_COMPLETION_LENGTH,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    try:
        return MolForgeAction(**extract_json(text)), text, ""
    except Exception as exc:
        return None, text, f"{exc.__class__.__name__}: {exc}"


def evaluate_policy_rollouts(model, tokenizer, label: str, output_path: Path) -> dict[str, Any]:
    previous_randomization = os.environ.get("MOLFORGE_TRAINING_RANDOMIZATION")
    previous_reward_mode = os.environ.get("MOLFORGE_REWARD_MODE")
    os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
    os.environ["MOLFORGE_REWARD_MODE"] = "assay_gated"
    if FastLanguageModel is not None:
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass
    model.eval()
    episodes: list[dict[str, Any]] = []
    try:
        env = MolForgeEnvironment()
        for _ in range(3):
            observation = env.reset()
            trace: list[dict[str, Any]] = []
            error = ""
            for _turn in range(EVAL_MAX_TURNS):
                if observation.done:
                    break
                action, raw_text, parse_error = generate_policy_action(model, tokenizer, observation)
                if action is None:
                    error = parse_error
                    trace.append(
                        {
                            "step": observation.step_index,
                            "raw_text": raw_text[:2000],
                            "error": parse_error,
                        }
                    )
                    break
                action = attach_team_messages(observation, attach_reasoning_fields(observation, action))
                observation = env.step(action)
                trace.append(
                    {
                        "step": observation.step_index,
                        "action_type": action.action_type,
                        "acting_role": action.acting_role,
                        "slot": action.slot,
                        "fragment": action.fragment,
                        "tool_name": action.tool_name,
                        "reward": observation.reward,
                        "governance_status": observation.governance.status,
                        "summary": observation.last_transition_summary,
                    }
                )
            scores = observation.metadata.get("terminal_grader_scores", {})
            episodes.append(
                {
                    "scenario_id": observation.scenario_id,
                    "difficulty": observation.difficulty,
                    "final_score": float(scores.get("final_score", scores.get("submission_score", 0.0))),
                    "submission_score": float(scores.get("submission_score", 0.0)),
                    "progress_score": float(scores.get("progress_score", 0.0)),
                    "steps": observation.step_index,
                    "done": observation.done,
                    "error": error,
                    "trace": trace,
                }
            )
    finally:
        if previous_randomization is None:
            os.environ.pop("MOLFORGE_TRAINING_RANDOMIZATION", None)
        else:
            os.environ["MOLFORGE_TRAINING_RANDOMIZATION"] = previous_randomization
        if previous_reward_mode is None:
            os.environ.pop("MOLFORGE_REWARD_MODE", None)
        else:
            os.environ["MOLFORGE_REWARD_MODE"] = previous_reward_mode
        if FastLanguageModel is not None:
            try:
                FastLanguageModel.for_training(model)
            except Exception:
                pass
        model.train()

    result = {
        "label": label,
        "episodes": episodes,
        "average_final_score": sum(row["final_score"] for row in episodes) / max(len(episodes), 1),
        "average_submission_score": sum(row["submission_score"] for row in episodes) / max(len(episodes), 1),
        "submit_rate": sum(1 for row in episodes if row["submission_score"] > 0.0) / max(len(episodes), 1),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def copy_outputs_to_drive() -> None:
    if not DRIVE_OUTPUT_DIR:
        return
    destination = Path(DRIVE_OUTPUT_DIR) / RUN_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(OUTPUT_DIR, destination)
    print(f"Copied RL outputs to {destination}")


def zip_outputs() -> Path:
    zip_base = OUTPUT_ROOT / RUN_NAME
    archive = shutil.make_archive(str(zip_base), "zip", OUTPUT_DIR)
    return Path(archive)


def main() -> None:
    manifest = {
        "run_name": RUN_NAME,
        "root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "base_model": BASE_MODEL_NAME,
        "sft_adapter_path": SFT_ADAPTER_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "rl_max_steps": RL_MAX_STEPS,
        "num_generations": NUM_GENERATIONS,
        "learning_rate": LEARNING_RATE,
        "per_device_batch": PER_DEVICE_BATCH,
        "gradient_accumulation": GRAD_ACCUM,
        "reward_mode": os.getenv("MOLFORGE_REWARD_MODE"),
        "drive_output_dir": DRIVE_OUTPUT_DIR,
        "run_eval": RUN_EVAL,
    }
    RUN_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    prompt_records = build_prompt_records()
    (OUTPUT_DIR / "prompt_dataset_preview.json").write_text(
        json.dumps(prompt_records[:20], indent=2),
        encoding="utf-8",
    )
    dataset = Dataset.from_list(prompt_records)
    print(f"RL prompt dataset: {len(dataset)} states")

    model, tokenizer = load_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if RUN_EVAL:
        before = evaluate_policy_rollouts(model, tokenizer, "before_rl", EVAL_BEFORE_JSON)
        print("Before-RL evaluation:", json.dumps(before, indent=2)[:3000])

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=molforge_reward_func,
        args=make_grpo_config(),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[JsonMetricsCallback()],
    )
    trainer.train()

    trainer.save_model(str(ADAPTER_SAVE_DIR))
    tokenizer.save_pretrained(str(ADAPTER_SAVE_DIR))
    if RUN_EVAL:
        after = evaluate_policy_rollouts(model, tokenizer, "after_rl", EVAL_AFTER_JSON)
        print("After-RL evaluation:", json.dumps(after, indent=2)[:3000])
    write_summary_and_plots()
    archive = zip_outputs()
    copy_outputs_to_drive()
    print(f"Saved RL adapter/logs to {OUTPUT_DIR}")
    print(f"Saved zip archive to {archive}")


if __name__ == "__main__":
    main()
