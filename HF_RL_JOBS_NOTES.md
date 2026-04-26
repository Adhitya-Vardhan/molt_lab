# Hugging Face RL Jobs Notes

This file tracks the remote RL training attempts for the MolForge OpenEnv GRPO run.

## Jobs Tried

| Job | Hardware | Result | Notes |
| --- | --- | --- | --- |
| `69ed7260d70108f37acdf4b8` | `a100-large` | Canceled | Stayed in `SCHEDULING`, so we canceled it before it used GPU time. |
| `69ed73d3d70108f37acdf4e1` | `l40sx1` | Failed | Started but exited during Python import before model load or training. |
| `69ed74f6d70108f37acdf504` | `l40sx1` | **Failed** | `--with mergekit` caused unsolvable pydantic conflict with `openenv-core`. |
| `69ed7be5d2c8bd8662bcef00` | `l40sx1` | Canceled | Incorrect CLI usage (missing image name). |
| `69ed9440d70108f37acdf83b` | `l40sx1` | Failed | `uv run` couldn't find the script path `issue/script.py`. |
| `69ed94add2c8bd8662bcf215` | `l40sx1` | Submitted | Fixed script path to just filename and used explicit `python` call. |

## Failure History

### Job 2 (`69ed73d3`) — `ModuleNotFoundError: No module named 'mergekit'`

TRL internally imports `mergekit` for GRPO model-merging callbacks even though we don't use merging. The fix was to add `--with mergekit`.

### Job 3 (`69ed74f6`) — **pydantic version conflict** (CURRENT)

Adding `--with mergekit` broke the resolver:

- `mergekit` (all versions) requires `pydantic < 2.11`
- `openenv-core==0.2.3` → `fastmcp>=3.0.0` → `pydantic >= 2.11.7`

**No version of pydantic satisfies both.** uv correctly refuses to resolve.

## Fix

**Do NOT pass `--with mergekit`** in the HF Jobs command. Instead, the script now installs mergekit at runtime with `--no-deps` before importing TRL:

```python
try:
    import mergekit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mergekit", "--no-deps", "-q"])
```

This makes `mergekit` importable (satisfying TRL) without pulling in its conflicting pydantic constraint.

## Checkpoint and Artifact Persistence

The OpenEnv GRPO script saves the final trained adapter and tokenizer to:

```text
<run_dir>/adapters/
```

It also writes logs, metrics, plots, before/after evaluator JSON, and a zip archive under the run directory. When `HF_OUTPUT_REPO=Adhitya122/molforge-rl-runs` is set, the full run folder is uploaded to:

```text
hf://datasets/Adhitya122/molforge-rl-runs/<run_name>
```

## Safer Next Runs

Recommended next HF Jobs command (NO `--with mergekit`):

```bash
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
