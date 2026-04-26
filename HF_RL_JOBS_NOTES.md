# Hugging Face RL Jobs Notes

This file tracks the remote RL training attempts for the MolForge OpenEnv GRPO run.

## Jobs Tried

| Job | Hardware | Result | Notes |
| --- | --- | --- | --- |
| `69ed7260d70108f37acdf4b8` | `a100-large` | Canceled | Stayed in `SCHEDULING`, so we canceled it before it used GPU time. |
| `69ed73d3d70108f37acdf4e1` | `l40sx1` | Failed | Started but exited during Python import before model load or training. |
| `69ed74f6d70108f37acdf504` | `l40sx1` | Running/scheduling | Smoke retry with `mergekit`, shorter completion length, fewer steps, and before/after eval skipped. |

## Current Failure

The `l40sx1` job failed before training started:

```text
RuntimeError: Failed to import trl.trainer.grpo_trainer ...
ModuleNotFoundError: No module named 'mergekit'
```

This is a dependency issue in the HF Jobs environment. Recent TRL imports GRPO callbacks that import `mergekit`, even though we are not explicitly merging models in our script.

## Fix

Include `mergekit` in the HF Jobs and Colab dependency list:

```bash
--with mergekit
```

The Colab notebook setup command now includes `mergekit`.

## Checkpoint and Artifact Persistence

The OpenEnv GRPO script saves the final trained adapter and tokenizer to:

```text
<run_dir>/adapters/
```

It also writes logs, metrics, plots, before/after evaluator JSON, and a zip archive under the run directory. When `HF_OUTPUT_REPO=Adhitya122/molforge-rl-runs` is set, the full run folder is uploaded to:

```text
hf://datasets/Adhitya122/molforge-rl-runs/<run_name>
```

For the failed `l40sx1` run, no checkpoint was created because the script failed at import time before it reached model loading.

## Safer Next Runs

Recommended next HF Jobs command changes:

```bash
--with mergekit
--env RL_MAX_STEPS=20
--env RL_DATASET_SIZE=30
--env MAX_COMPLETION_LENGTH=1024
```

Use this as a smoke run first. Once it reaches at least one trainer log line and uploads artifacts, scale back to:

```bash
--env RL_MAX_STEPS=80
--env RL_DATASET_SIZE=120
--env MAX_COMPLETION_LENGTH=2048
```

Good hardware choices:

| Hardware | Use |
| --- | --- |
| `l40sx1` | Best next smoke test: 48 GB VRAM, cheaper than A100. |
| `a100-large` | Good full run if scheduling is available. |
| `h200` | Highest headroom, more expensive, useful if A100 scheduling stalls. |
| `a10g-large` | Cheap fallback, but may need shorter completion length and fewer steps. |

## Monitoring Commands

```bash
hf jobs inspect <job_id>
hf jobs logs <job_id> --tail 200
```

Use logs without `inspect` when searching for the real traceback, because `inspect` prints the full base64-encoded submitted script and makes the useful error harder to see.
