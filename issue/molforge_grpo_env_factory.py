# -*- coding: utf-8 -*-
"""MolForge GRPO training with environment_factory (official TRL pattern).

This script trains a model to play the MolForge drug-design environment using
TRL's ``environment_factory`` pattern — the same approach used in the official
Sudoku, Wordle, and Echo examples. The trainer automatically handles:

1. Creating environment instances per rollout
2. Generating model completions and parsing tool calls
3. Stepping through the environment with the model's actions
4. Collecting rewards and managing the interaction loop

Outputs:
- ``{OUTPUT_DIR}/logs/trainer_log.jsonl``  — per-step trainer metrics
- ``{OUTPUT_DIR}/logs/run_manifest.json``  — config snapshot
- ``{OUTPUT_DIR}/plots/reward_curve.png``  — reward over training steps
- ``{OUTPUT_DIR}/plots/loss_curve.png``    — training loss curve
- ``{OUTPUT_DIR}/plots/reward_signals.png``— per-signal reward breakdown
- ``{OUTPUT_DIR}/rl_summary.json``         — aggregate stats

Usage on Colab (T4 GPU):

    !pip install -U "trl[vllm]" pydantic datasets matplotlib
    !python issue/molforge_grpo_env_factory.py

Or with Unsloth for 2x memory efficiency:

    !pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install -U "trl>=0.21.0" pydantic datasets matplotlib
    !python issue/molforge_grpo_env_factory.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Bootstrap openenv shim if needed ──────────────────────────────────────
try:
    import openenv  # noqa: F401
except ImportError:
    import openenv_shim  # noqa: F401

# ── Core imports ──────────────────────────────────────────────────────────
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from molforge_tool_env import (
    MolForgeToolEnv,
    reward_environment,
    reward_valid_actions,
    reward_progress,
    reward_submission,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "200"))
MAX_STEPS = int(os.getenv("RL_MAX_STEPS", "50"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "4096"))
LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "5e-6"))
GRAD_ACCUM = int(os.getenv("RL_GRAD_ACCUM", "16"))
SAVE_STEPS = int(os.getenv("RL_SAVE_STEPS", "10"))
TEMPERATURE = float(os.getenv("RL_TEMPERATURE", "0.8"))
SEED = int(os.getenv("RL_SEED", "3407"))
USE_VLLM = os.getenv("USE_VLLM", "true").lower() in {"1", "true", "yes"}
PUSH_TO_HUB = os.getenv("PUSH_TO_HUB", "false").lower() in {"1", "true", "yes"}
RUN_NAME = os.getenv("RUN_NAME", time.strftime("molforge_grpo_%Y%m%d_%H%M%S"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"molforge-grpo-{RUN_NAME}")

# Reward mode: "curriculum" for warm-up, "assay_gated" for final eval
os.environ.setdefault("MOLFORGE_REWARD_MODE", "curriculum")
os.environ.setdefault("MOLFORGE_TRAINING_RANDOMIZATION", "1")

# Output directories
_OUTPUT_PATH = Path(OUTPUT_DIR)
_LOG_DIR = _OUTPUT_PATH / "logs"
_PLOT_DIR = _OUTPUT_PATH / "plots"
for _d in [_OUTPUT_PATH, _LOG_DIR, _PLOT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

_TRAINER_LOG = _LOG_DIR / "trainer_log.jsonl"
_RUN_MANIFEST = _LOG_DIR / "run_manifest.json"
_SUMMARY_JSON = _OUTPUT_PATH / "rl_summary.json"


# ═══════════════════════════════════════════════════════════════════════════
#  Logging callback — writes per-step metrics to JSONL
# ═══════════════════════════════════════════════════════════════════════════

class MetricsLogger(TrainerCallback):
    """Log every trainer step to JSONL and generate plots on save."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            row = {"step": state.global_step, "time": time.time(), **logs}
            with _TRAINER_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def on_save(self, args, state, control, **kwargs):
        try:
            generate_plots()
        except Exception as exc:
            print(f"Plot generation skipped: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot generation — reward curves, loss curves, signal breakdown
# ═══════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def generate_plots() -> None:
    """Generate matplotlib plots from trainer log history."""
    rows = _read_jsonl(_TRAINER_LOG)
    if not rows:
        print("No log data yet, skipping plots.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots.")
        return

    # ── 1. Reward curve ────────────────────────────────────────────────
    reward_rows = [(r["step"], r["reward"]) for r in rows if "reward" in r]
    if reward_rows:
        steps, rewards = zip(*reward_rows)
        window = max(4, len(rewards) // 10)
        moving_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(sum(rewards[start:i+1]) / (i - start + 1))

        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards, alpha=0.3, label="step reward", color="#4A90D9")
        plt.plot(steps, moving_avg, label=f"moving avg ({window})", color="#E74C3C", linewidth=2)
        plt.xlabel("Training Step")
        plt.ylabel("Reward")
        plt.title("MolForge GRPO — Reward Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_PLOT_DIR / "reward_curve.png", dpi=160)
        plt.close()
        print(f"  📊 Saved {_PLOT_DIR / 'reward_curve.png'}")

    # ── 2. Loss curve ──────────────────────────────────────────────────
    loss_rows = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    if loss_rows:
        steps, losses = zip(*loss_rows)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#2ECC71")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("MolForge GRPO — Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_PLOT_DIR / "loss_curve.png", dpi=160)
        plt.close()
        print(f"  📊 Saved {_PLOT_DIR / 'loss_curve.png'}")

    # ── 3. Per-signal reward breakdown ─────────────────────────────────
    signal_keys = [
        ("reward_reward_environment", "Environment", "#3498DB"),
        ("reward_reward_valid_actions", "Valid Actions", "#E67E22"),
        ("reward_reward_progress", "Progress", "#9B59B6"),
        ("reward_reward_submission", "Submission", "#1ABC9C"),
    ]
    has_signals = any(k in rows[-1] for k, _, _ in signal_keys)
    if has_signals:
        plt.figure(figsize=(10, 5))
        for key, label, color in signal_keys:
            signal_rows = [(r["step"], r[key]) for r in rows if key in r]
            if signal_rows:
                s, v = zip(*signal_rows)
                plt.plot(s, v, label=label, color=color, linewidth=1.5, alpha=0.8)
        plt.xlabel("Training Step")
        plt.ylabel("Reward Signal")
        plt.title("MolForge GRPO — Reward Signals Breakdown")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_PLOT_DIR / "reward_signals.png", dpi=160)
        plt.close()
        print(f"  📊 Saved {_PLOT_DIR / 'reward_signals.png'}")

    # ── 4. Grad norm ───────────────────────────────────────────────────
    grad_rows = [(r["step"], r["grad_norm"]) for r in rows if "grad_norm" in r]
    if grad_rows:
        steps, norms = zip(*grad_rows)
        plt.figure(figsize=(10, 4))
        plt.plot(steps, norms, marker="o", markersize=3, linewidth=1.5, color="#E74C3C")
        plt.xlabel("Training Step")
        plt.ylabel("Gradient Norm")
        plt.title("MolForge GRPO — Gradient Norm")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_PLOT_DIR / "grad_norm.png", dpi=160)
        plt.close()
        print(f"  📊 Saved {_PLOT_DIR / 'grad_norm.png'}")

    # ── Summary JSON ───────────────────────────────────────────────────
    all_rewards = [r.get("reward", 0) for r in rows if "reward" in r]
    summary = {
        "run_name": RUN_NAME,
        "model": MODEL_NAME,
        "reward_mode": os.getenv("MOLFORGE_REWARD_MODE"),
        "total_steps": len(rows),
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1) if all_rewards else 0,
        "max_reward": max(all_rewards) if all_rewards else 0,
        "min_reward": min(all_rewards) if all_rewards else 0,
        "final_loss": loss_rows[-1][1] if loss_rows else None,
    }
    _SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  📄 Saved {_SUMMARY_JSON}")


# ═══════════════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert medicinal chemist designing kinase inhibitor drug candidates.

## YOUR TASK

Design a molecule that meets the scenario's potency, toxicity, and synthesizability constraints
within a limited oracle budget. You work by editing a modular molecule (4 structural slots) and
running computational assays to gather evidence before submitting your final candidate.

## AVAILABLE TOOLS

You have 5 tools to interact with the environment:

1. **edit_molecule(slot, fragment, edit_type, rationale)** — Edit a molecular slot.
   - slot: warhead, hinge, solvent_tail, or back_pocket
   - fragment: the fragment identifier to place
   - edit_type: add_fragment, substitute, remove, or undo_last_edit
   - rationale: why this edit should improve the molecule

2. **run_assay(tool_name, rationale)** — Run a computational assay.
   - tool_name: evaluate_properties, dock_target, assay_toxicity, estimate_synthesizability,
     evaluate_novelty, search_literature, or run_md_simulation
   - rationale: why this assay is needed now

3. **submit_molecule(rationale)** — Submit the current molecule as the final candidate.
   - Only submit when you have sufficient evidence that constraints are met.

4. **restart_episode(rationale)** — Restart with a fresh scaffold.
   - Use when the current molecular series is fundamentally flawed.

5. **defer_action(rationale)** — Skip this turn.
   - Use when no productive action is available.

## STRATEGY

1. First, run assays to understand the starting molecule's properties.
2. Make targeted edits to improve weak properties.
3. Re-run assays to verify improvements.
4. Submit when evidence shows all constraints are satisfied.
5. Be budget-conscious — each assay costs oracle budget points.

## CONSTRAINTS

- You have a limited oracle budget — don't waste it on redundant assays.
- You have a limited number of steps before the episode terminates.
- Invalid actions (wrong fragments, vetoed edits) cost you reward.
- The governance board may veto unsafe edits — adjust your strategy accordingly.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

def make_dataset(size: int) -> Dataset:
    """Create a dataset of repeated prompts to control training episodes."""
    return Dataset.from_dict({
        "prompt": [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Design a drug candidate that meets the scenario constraints. Use the available tools to edit the molecule, run assays, and submit when ready."},
        ]] * size
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    manifest = {
        "run_name": RUN_NAME,
        "model": MODEL_NAME,
        "dataset_size": DATASET_SIZE,
        "max_steps": MAX_STEPS,
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "learning_rate": LEARNING_RATE,
        "grad_accum": GRAD_ACCUM,
        "temperature": TEMPERATURE,
        "use_vllm": USE_VLLM,
        "output_dir": OUTPUT_DIR,
        "reward_mode": os.getenv("MOLFORGE_REWARD_MODE"),
        "seed": SEED,
    }
    _RUN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{'='*60}")
    print(f" MolForge GRPO Training (environment_factory)")
    print(f"{'='*60}")
    print(json.dumps(manifest, indent=2))
    print(f"{'='*60}")

    dataset = make_dataset(DATASET_SIZE)
    print(f"Dataset: {len(dataset)} episodes")

    # ── GRPOConfig ─────────────────────────────────────────────────────
    grpo_config_kwargs = {
        # Training schedule
        "num_train_epochs": 1,
        "learning_rate": LEARNING_RATE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "per_device_train_batch_size": 1,
        "warmup_steps": 10,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "max_steps": MAX_STEPS,

        # GRPO
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "log_completions": True,
        "num_completions_to_print": 1,
        "chat_template_kwargs": {"enable_thinking": False},

        # Logging
        "output_dir": OUTPUT_DIR,
        "logging_steps": 1,
        "save_steps": SAVE_STEPS,
        "save_total_limit": 2,

        # Sampling
        "temperature": TEMPERATURE,
        "top_k": 10,

        # Seed
        "seed": SEED,
    }

    # Add vLLM config if available
    if USE_VLLM:
        grpo_config_kwargs.update({
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": 0.15,
        })

    if PUSH_TO_HUB:
        grpo_config_kwargs["push_to_hub"] = True

    grpo_config = GRPOConfig(**grpo_config_kwargs)

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            reward_environment,
            reward_valid_actions,
            reward_progress,
            reward_submission,
        ],
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=MolForgeToolEnv,
        callbacks=[MetricsLogger()],
    )

    # ── GPU stats ──────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            print(f"{start_gpu_memory} GB of memory reserved.")
    except Exception:
        pass

    # ── Train ──────────────────────────────────────────────────────────
    print("\nStarting GRPO training with MolForge environment...")
    trainer_stats = trainer.train()

    # ── Save ───────────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    print(f"\nSaved model to {OUTPUT_DIR}")

    if PUSH_TO_HUB:
        trainer.push_to_hub()
        print("Pushed model to Hugging Face Hub.")

    # ── Generate final plots & summary ─────────────────────────────────
    print("\nGenerating reward curves and plots...")
    generate_plots()

    # ── Final stats ────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            print(f"\nTraining time: {trainer_stats.metrics['train_runtime']:.1f}s")
            print(f"Peak GPU memory: {used_memory} GB")
    except Exception:
        pass

    # ── Copy to Drive (Colab only) ─────────────────────────────────────
    drive_output = os.getenv("DRIVE_OUTPUT_DIR", "")
    if drive_output:
        import shutil
        dest = Path(drive_output) / RUN_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(_OUTPUT_PATH, dest)
        print(f"Copied outputs to {dest}")

    print(f"\n{'='*60}")
    print(f" ✅ MolForge GRPO training complete!")
    print(f" 📊 Plots saved to:  {_PLOT_DIR}")
    print(f" 📄 Logs saved to:   {_LOG_DIR}")
    print(f" 🧬 Model saved to:  {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
