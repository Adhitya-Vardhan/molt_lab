# -*- coding: utf-8 -*-
"""OpenEnv-style GRPO training for MolForge.

This follows the TRL OpenEnv examples: GRPOTrainer receives an
environment_factory, public environment methods become model-callable tools,
and reward functions read rewards from the environment instances.

The older qwen3_5_2b_unsloth_grpo_rl.py script trains on single JSON actions.
This script is the judge-facing OpenEnv tool-loop variant.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_URL = os.getenv("MOLFORGE_REPO_URL", "https://github.com/Adhitya-Vardhan/molt_lab.git")
DEFAULT_PROJECT_ROOT = "/content/molt_lab" if Path("/content").exists() else "/kaggle/working/molt_lab"
PROJECT_ROOT = Path(os.getenv("MOLFORGE_PROJECT_ROOT", DEFAULT_PROJECT_ROOT))


def ensure_project_available() -> Path:
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
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Skipping Google Drive mount: {exc}")

# ── Pre-import fix: install mergekit WITHOUT its deps ──────────────
# TRL internally imports mergekit for model-merging callbacks, but we
# don't use merging.  mergekit's pydantic constraint (<2.11) conflicts
# with openenv-core's fastmcp (>=2.11.7).  Installing --no-deps makes
# the module importable without pulling in the conflicting pydantic pin.
try:
    import mergekit  # noqa: F401
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "mergekit", "--no-deps", "-q"],
    )

import torch  # noqa: E402

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
except Exception:  # pragma: no cover
    FastLanguageModel = None

    def is_bfloat16_supported() -> bool:
        return False

from datasets import Dataset  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

try:
    from trl import RichProgressCallback  # noqa: E402
except Exception:  # pragma: no cover
    RichProgressCallback = None

from inference_common import attach_reasoning_fields, attach_team_messages  # noqa: E402
from models import MolForgeAction  # noqa: E402
from server.molforge_environment import MolForgeEnvironment  # noqa: E402


BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/Qwen3.5-2B")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
RL_MAX_STEPS = int(os.getenv("RL_MAX_STEPS", "80"))
DATASET_SIZE = int(os.getenv("RL_DATASET_SIZE", "120"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
PER_DEVICE_BATCH = int(os.getenv("RL_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("RL_GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "2e-6"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "2048"))
LOGGING_STEPS = int(os.getenv("RL_LOGGING_STEPS", "1"))
SAVE_STEPS = int(os.getenv("RL_SAVE_STEPS", "25"))
SEED = int(os.getenv("RL_SEED", "3407"))
USE_UNSLOTH = os.getenv("USE_UNSLOTH", "true").lower() in {"1", "true", "yes"}
RUN_BEFORE_EVAL = os.getenv("RUN_BEFORE_EVAL", "true").lower() in {"1", "true", "yes"}
RUN_AFTER_EVAL = os.getenv("RUN_AFTER_EVAL", "true").lower() in {"1", "true", "yes"}
RUN_NAME = os.getenv("RUN_NAME", time.strftime("molforge_openenv_grpo_%Y%m%d_%H%M%S"))
DEFAULT_OUTPUT_ROOT = "/content/molforge_rl_runs" if Path("/content").exists() else "/kaggle/working/molforge_rl_runs"
OUTPUT_ROOT = Path(os.getenv("RL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
ADAPTER_SAVE_DIR = OUTPUT_DIR / "adapters"
DRIVE_OUTPUT_DIR = os.getenv("DRIVE_OUTPUT_DIR", "")
HF_OUTPUT_REPO = os.getenv("HF_OUTPUT_REPO", "")
HF_OUTPUT_REPO_TYPE = os.getenv("HF_OUTPUT_REPO_TYPE", "dataset")

for path in [OUTPUT_DIR, LOG_DIR, PLOT_DIR, ADAPTER_SAVE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TOOL_LOG = LOG_DIR / "openenv_tool_rollouts.jsonl"
TRAINER_LOG = LOG_DIR / "trainer_log_history.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "openenv_rl_summary.json"
METRICS_CSV = OUTPUT_DIR / "openenv_tool_metrics.csv"
EVAL_BEFORE_JSON = OUTPUT_DIR / "eval_before_training.json"
EVAL_AFTER_JSON = OUTPUT_DIR / "eval_after_training.json"


def find_sft_adapter_path() -> str:
    explicit = os.getenv("SFT_ADAPTER_PATH")
    if explicit:
        return explicit
    candidates = [
        Path("/content/molforge_space/adapters/qwen3_5_2b_lora_adapters_compact_v4"),
        Path("/content/drive/MyDrive/Qwen_3.5_finetune/qwen3_5_2b_lora_adapters_compact_v4"),
        Path("/kaggle/working/qwen3_5_2b_lora_adapters_compact_v4"),
    ]
    for root in [Path("/kaggle/input"), Path("/content")]:
        if root.exists():
            candidates.extend(sorted(root.rglob("qwen3_5_2b_lora_adapters_compact_v4")))
            candidates.extend(sorted(root.rglob("adapter_config.json")))
    for candidate in candidates:
        if candidate.name == "adapter_config.json":
            candidate = candidate.parent
        if (candidate / "adapter_config.json").exists():
            return str(candidate)
    raise FileNotFoundError("Set SFT_ADAPTER_PATH to the v4 LoRA adapter directory.")


SFT_ADAPTER_PATH = find_sft_adapter_path()


def observation_text(obs) -> str:
    assays = ", ".join(f"{a.property_name}={a.estimate:.3f}" for a in obs.known_assays[-5:]) or "none"
    constraints = "; ".join(
        f"{c.name}:{c.evidence_status}:{c.actual if c.actual is not None else 'unknown'}" for c in obs.constraint_status
    )
    return (
        f"Scenario: {obs.scenario_id} ({obs.difficulty})\n"
        f"Step: {obs.step_index}/{obs.max_steps}; budget: {obs.remaining_budget}/{obs.max_budget}\n"
        f"Goal: {obs.task_brief}\n"
        f"Molecule: {obs.current_molecule}\n"
        f"Known assays: {assays}\n"
        f"Constraints: {constraints}\n"
        "Use the MolForge tools to edit, run assays, restart, defer, or submit."
    )


def terminal_scores(obs) -> dict[str, float]:
    scores = obs.metadata.get("terminal_grader_scores", {}) if obs.done else {}
    return {
        "final_score": float(scores.get("final_score", 0.0)),
        "submission_score": float(scores.get("submission_score", 0.0)),
        "progress_score": float(scores.get("progress_score", 0.0)),
    }


class MolForgeToolEnv:
    def __init__(self):
        self.env = MolForgeEnvironment()
        self.observation = None
        self.reward = 0.0
        self.final_score = 0.0
        self.submission_score = 0.0
        self.progress_score = 0.0
        self.done = False
        self.tool_calls: list[dict[str, Any]] = []

    def reset(self, **kwargs) -> str:
        scenario_index = int(kwargs.get("scenario_index", 0))
        self.env = MolForgeEnvironment()
        self.reward = 0.0
        self.final_score = 0.0
        self.submission_score = 0.0
        self.progress_score = 0.0
        self.done = False
        self.tool_calls = []
        for _ in range(scenario_index + 1):
            self.observation = self.env.reset()
        return observation_text(self.observation)

    def _apply(self, action: MolForgeAction) -> str:
        if self.done:
            raise ValueError("Episode is already done.")
        action = attach_team_messages(self.observation, attach_reasoning_fields(self.observation, action))
        self.observation = self.env.step(action)
        scores = terminal_scores(self.observation)
        self.final_score = scores["final_score"]
        self.submission_score = scores["submission_score"]
        self.progress_score = scores["progress_score"]
        transition_reward = float(self.observation.reward)
        self.reward = self.final_score if self.observation.done else transition_reward
        self.done = bool(self.observation.done)
        row = {
            "timestamp": time.time(),
            "scenario_id": self.observation.scenario_id,
            "difficulty": self.observation.difficulty,
            "step_index": self.observation.step_index,
            "action_type": action.action_type,
            "acting_role": action.acting_role,
            "slot": action.slot,
            "fragment": action.fragment,
            "tool_name": action.tool_name,
            "reward": self.reward,
            "transition_reward": transition_reward,
            "done": self.done,
            "governance_status": self.observation.governance.status,
            "final_score": self.final_score,
            "submission_score": self.submission_score,
            "progress_score": self.progress_score,
            "summary": self.observation.last_transition_summary,
        }
        self.tool_calls.append(row)
        with TOOL_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return observation_text(self.observation) + f"\nLast result: {self.observation.last_transition_summary}"

    def edit(self, slot: str, fragment: str, rationale: str = "Improve the molecule.") -> str:
        """Edit one molecular fragment slot.

        Args:
            slot: One of warhead, hinge, solvent_tail, back_pocket.
            fragment: Fragment name to place in that slot.
            rationale: Short reason for the edit.

        Returns:
            Updated MolForge observation after the edit.
        """
        return self._apply(
            MolForgeAction(
                action_type="edit",
                acting_role="lead_chemist",
                edit_type="substitute",
                slot=slot,
                fragment=fragment,
                rationale=rationale,
            )
        )

    def run_assay(self, tool_name: str, rationale: str = "Collect missing evidence.") -> str:
        """Run one MolForge assay or oracle tool.

        Args:
            tool_name: One of evaluate_properties, dock_target, assay_toxicity, estimate_synthesizability, evaluate_novelty, search_literature, run_md_simulation.
            rationale: Short reason for running the assay.

        Returns:
            Updated MolForge observation after the assay.
        """
        return self._apply(
            MolForgeAction(
                action_type="run_assay",
                acting_role="assay_planner",
                tool_name=tool_name,
                rationale=rationale,
            )
        )

    def submit(self, rationale: str = "Submit the evidence-supported candidate.") -> str:
        """Submit the current molecule for final grading.

        Args:
            rationale: Short reason the evidence is sufficient.

        Returns:
            Final MolForge report card and scores.
        """
        return self._apply(MolForgeAction(action_type="submit", acting_role="lead_chemist", rationale=rationale))

    def restart(self, rationale: str = "Restart from the safer alternate series.") -> str:
        """Restart the molecule design episode.

        Args:
            rationale: Short reason restart is better than continuing.

        Returns:
            Updated MolForge observation after restart.
        """
        return self._apply(MolForgeAction(action_type="restart", acting_role="lead_chemist", rationale=rationale))

    def defer(self, rationale: str = "No reliable action is available.") -> str:
        """Defer when no useful action is available.

        Args:
            rationale: Short reason for deferring.

        Returns:
            Updated MolForge observation after defer.
        """
        return self._apply(MolForgeAction(action_type="defer", acting_role="lead_chemist", rationale=rationale))


def reward_final(environments, **kwargs) -> list[float]:
    return [float(env.final_score if env.done else max(env.reward, 0.0)) for env in environments]


def reward_submit(environments, **kwargs) -> list[float]:
    return [float(env.submission_score) for env in environments]


def reward_valid_progress(environments, **kwargs) -> list[float]:
    rewards = []
    for env in environments:
        if not env.tool_calls:
            rewards.append(-0.2)
            continue
        latest = env.tool_calls[-1]
        penalty = -0.2 if latest["governance_status"] == "policy_veto" else 0.0
        rewards.append(float(latest["transition_reward"]) + penalty)
    return rewards


def build_dataset() -> Dataset:
    prompt = (
        "You are the MolForge medicinal chemistry agent. "
        "Use the provided tools to improve potency, collect safety and synthesis evidence, "
        "respect the assay budget, and submit only when the candidate is evidence-supported."
    )
    rows = []
    for index in range(DATASET_SIZE):
        scenario_index = index % 3
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "scenario_index": scenario_index,
                "scenario_name": ["level_0_easy", "level_1_medium", "level_2_hard"][scenario_index],
            }
        )
    return Dataset.from_list(rows)


class JsonMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
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
            
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer
                
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
    
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        
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
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "temperature": float(os.getenv("RL_TEMPERATURE", "0.7")),
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "report_to": os.getenv("REPORT_TO", "none"),
        "log_completions": True,
        "num_completions_to_print": 1,
        "remove_unused_columns": False,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    signature = inspect.signature(GRPOConfig.__init__).parameters
    return GRPOConfig(**{key: value for key, value in kwargs.items() if key in signature})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_tool_metrics_csv() -> list[dict[str, Any]]:
    rows = read_jsonl(TOOL_LOG)
    if not rows:
        return []
    fieldnames = [
        "timestamp",
        "scenario_id",
        "difficulty",
        "step_index",
        "action_type",
        "slot",
        "fragment",
        "tool_name",
        "reward",
        "transition_reward",
        "done",
        "governance_status",
        "final_score",
        "submission_score",
        "progress_score",
        "summary",
    ]
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_series(rows: list[dict[str, Any]], metric: str, output_path: Path, xlabel: str) -> None:
    points = [(idx, row.get(metric)) for idx, row in enumerate(rows) if row.get(metric) is not None]
    if not points:
        return
    xs, ys = zip(*points)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys, alpha=0.35, label=metric)
    window = max(4, min(32, len(ys) // 10 or 4))
    moving = [sum(ys[max(0, i - window + 1) : i + 1]) / len(ys[max(0, i - window + 1) : i + 1]) for i in range(len(ys))]
    plt.plot(xs, moving, label=f"moving avg ({window})")
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_summary_and_plots() -> None:
    tool_rows = write_tool_metrics_csv()
    trainer_rows = read_jsonl(TRAINER_LOG)
    eval_before = read_json_file(EVAL_BEFORE_JSON)
    eval_after = read_json_file(EVAL_AFTER_JSON)
    rewards = [float(row.get("reward", 0.0)) for row in tool_rows]
    action_counts = Counter(row.get("action_type", "unknown") for row in tool_rows)
    summary = {
        "run_name": RUN_NAME,
        "base_model": BASE_MODEL_NAME,
        "sft_adapter_path": SFT_ADAPTER_PATH,
        "num_tool_calls": len(tool_rows),
        "mean_tool_reward": sum(rewards) / max(len(rewards), 1),
        "max_tool_reward": max(rewards) if rewards else 0.0,
        "action_counts": dict(action_counts),
        "latest_trainer_log": trainer_rows[-1] if trainer_rows else {},
        "eval_before": {
            "average_final_score": eval_before.get("average_final_score"),
        },
        "eval_after": {
            "average_final_score": eval_after.get("average_final_score"),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not tool_rows:
        return
    try:
        import matplotlib.pyplot as plt

        plot_series(tool_rows, "reward", PLOT_DIR / "reward_curve.png", "Tool call")
        plot_series(tool_rows, "final_score", PLOT_DIR / "final_score_curve.png", "Tool call")
        if trainer_rows:
            plot_series(trainer_rows, "loss", PLOT_DIR / "loss_curve.png", "Trainer step")
            plot_series(trainer_rows, "grad_norm", PLOT_DIR / "grad_norm_curve.png", "Trainer step")
        if eval_before and eval_after:
            plot_eval_comparison(eval_before, eval_after)
        labels, counts = zip(*action_counts.items())
        plt.figure(figsize=(8, 4))
        plt.bar(labels, counts)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "action_distribution.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def action_to_tool_call(action: MolForgeAction) -> tuple[str, dict[str, Any]]:
    if action.action_type == "edit":
        return "edit", {"slot": action.slot or "solvent_tail", "fragment": action.fragment or "morpholine"}
    if action.action_type == "run_assay":
        return "run_assay", {"tool_name": action.tool_name or "evaluate_properties"}
    if action.action_type == "submit":
        return "submit", {}
    if action.action_type == "restart":
        return "restart", {}
    return "defer", {}


def generate_json_policy_action(model, tokenizer, observation) -> MolForgeAction | None:
    from inference_common import COMPACT_SYSTEM_PROMPT, build_model_payload, extract_json

    messages = [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(build_model_payload(observation, compact=True), separators=(",", ":"))},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=384,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
    try:
        return MolForgeAction(**extract_json(text))
    except Exception:
        return None


def evaluate_json_policy_rollouts(model, tokenizer, output_path: Path, label: str) -> dict[str, Any]:
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
    episodes = []
    try:
        env = MolForgeEnvironment()
        for _ in range(3):
            obs = env.reset()
            trace = []
            for _turn in range(10):
                if obs.done:
                    break
                action = generate_json_policy_action(model, tokenizer, obs)
                if action is None:
                    trace.append({"step": obs.step_index, "error": "parse_failed"})
                    break
                action = attach_team_messages(obs, attach_reasoning_fields(obs, action))
                obs = env.step(action)
                trace.append({"step": obs.step_index, "action_type": action.action_type, "reward": obs.reward})
            scores = obs.metadata.get("terminal_grader_scores", {})
            episodes.append(
                {
                    "scenario_id": obs.scenario_id,
                    "difficulty": obs.difficulty,
                    "final_score": float(scores.get("final_score", scores.get("submission_score", 0.0))),
                    "submission_score": float(scores.get("submission_score", 0.0)),
                    "progress_score": float(scores.get("progress_score", 0.0)),
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


def upload_outputs_to_hub() -> None:
    if not HF_OUTPUT_REPO:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(HF_OUTPUT_REPO, repo_type=HF_OUTPUT_REPO_TYPE, exist_ok=True)
        api.upload_folder(
            repo_id=HF_OUTPUT_REPO,
            repo_type=HF_OUTPUT_REPO_TYPE,
            folder_path=str(OUTPUT_DIR),
            path_in_repo=RUN_NAME,
            commit_message=f"Upload MolForge OpenEnv GRPO run {RUN_NAME}",
        )
        print(f"Uploaded RL outputs to hf://{HF_OUTPUT_REPO_TYPE}s/{HF_OUTPUT_REPO}/{RUN_NAME}")
    except Exception as exc:
        print(f"HF output upload failed: {exc}")


def zip_outputs() -> Path:
    return Path(shutil.make_archive(str(OUTPUT_ROOT / RUN_NAME), "zip", OUTPUT_DIR))


def main() -> None:
    manifest = {
        "run_name": RUN_NAME,
        "script": "qwen3_5_2b_unsloth_grpo_openenv_tools.py",
        "root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "base_model": BASE_MODEL_NAME,
        "sft_adapter_path": SFT_ADAPTER_PATH,
        "rl_max_steps": RL_MAX_STEPS,
        "dataset_size": DATASET_SIZE,
        "num_generations": NUM_GENERATIONS,
        "batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "run_before_eval": RUN_BEFORE_EVAL,
        "run_after_eval": RUN_AFTER_EVAL,
        "drive_output_dir": DRIVE_OUTPUT_DIR,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    dataset = build_dataset()
    model, tokenizer = load_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if RUN_BEFORE_EVAL:
        before = evaluate_json_policy_rollouts(model, tokenizer, EVAL_BEFORE_JSON, "before_rl")
        print("Before-RL evaluator:", json.dumps(before, indent=2)[:3000])
    else:
        EVAL_BEFORE_JSON.write_text(
            json.dumps({"label": "before_rl", "skipped": True, "reason": "RUN_BEFORE_EVAL=false"}, indent=2),
            encoding="utf-8",
        )

    callbacks = [JsonMetricsCallback()]
    if RichProgressCallback is not None:
        callbacks.append(RichProgressCallback())
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_valid_progress, reward_final, reward_submit],
        args=make_grpo_config(),
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=MolForgeToolEnv,
        callbacks=callbacks,
    )
    try:
        trainer.train()
        trainer.save_model(str(ADAPTER_SAVE_DIR))
        tokenizer.save_pretrained(str(ADAPTER_SAVE_DIR))
        if RUN_AFTER_EVAL:
            after = evaluate_json_policy_rollouts(model, tokenizer, EVAL_AFTER_JSON, "after_rl")
            print("After-RL evaluator:", json.dumps(after, indent=2)[:3000])
        else:
            EVAL_AFTER_JSON.write_text(
                json.dumps({"label": "after_rl", "skipped": True, "reason": "RUN_AFTER_EVAL=false"}, indent=2),
                encoding="utf-8",
            )
    finally:
        write_summary_and_plots()
        archive = zip_outputs()
        copy_outputs_to_drive()
        upload_outputs_to_hub()
        print(f"Saved OpenEnv RL run to {OUTPUT_DIR}")
        print(f"Saved zip archive to {archive}")


if __name__ == "__main__":
    main()
