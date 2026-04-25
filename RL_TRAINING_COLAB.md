# MolForge RL Training in Colab

Use [issue/molforge_grpo_colab_training.ipynb](issue/molforge_grpo_colab_training.ipynb) for the judge-rerunnable workflow.

The notebook trains from the Qwen3.5 2B SFT v4 adapter with TRL GRPO against the real MolForge environment reward. It is set up for short evidence runs on A100/H100 rather than full convergence.

## Outputs

Each run writes to `/content/molforge_rl_runs/<run_name>/` and copies the same folder to `DRIVE_OUTPUT_DIR` when set.

Important artifacts:

- `logs/completion_rewards.jsonl`: every sampled completion, parsed action, reward, governance status, and score diagnostics.
- `logs/trainer_log_history.jsonl`: trainer loss, grad norm, learning rate, and step timing.
- `completion_metrics.csv`: spreadsheet-friendly completion reward table.
- `eval_before_training.json`: full 3-task rollout before GRPO.
- `eval_after_training.json`: full 3-task rollout after GRPO.
- `plots/reward_curve.png`: completion reward curve and moving average.
- `plots/loss_curve.png`: trainer loss curve.
- `plots/eval_before_after.png`: before/after final_score comparison.
- `plots/action_distribution.png`: sampled action mix.
- `adapters/`: trained LoRA adapter checkpoint.
- `<run_name>.zip`: portable archive of the run outputs.

## Fast Demo Settings

For a quick A100/H100 proof run:

```python
os.environ["RL_MAX_STEPS"] = "80"
os.environ["NUM_GENERATIONS"] = "2"
os.environ["RL_PROMPT_EPISODES"] = "96"
os.environ["RL_BATCH_SIZE"] = "2"
os.environ["RL_GRAD_ACCUM"] = "4"
os.environ["RL_LEARNING_RATE"] = "2e-6"
```

For a stronger run, try `RL_MAX_STEPS=200` and `NUM_GENERATIONS=4` on H100.
If Colab runs out of memory, reduce `MAX_PROMPT_LENGTH` to `1024`; keep `RL_BATCH_SIZE` divisible by `NUM_GENERATIONS`.

## What to Show Judges

Use the before/after rollout JSON plus these plots:

- `reward_curve.png` for reward improvement during RL.
- `loss_curve.png` for actual training evidence.
- `eval_before_after.png` for task-level behavior change.

The official environment score remains `final_score`; `progress_score` and per-step rewards are debugging signals.
