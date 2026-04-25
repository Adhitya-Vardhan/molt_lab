# MolForge Training Instructions

This guide is for training a small model against MolForge without teaching it to exploit the environment.

## 1. Safety Defaults

MolForge now hides true internal molecule properties from public `state()` metadata by default. If you need to debug the environment manually, use:

```bash
MOLFORGE_DEBUG_STATE=1 python inference.py
```

Do not use `MOLFORGE_DEBUG_STATE=1` while collecting SFT data or running RL.

The chemistry oracle path uses RDKit descriptors by default and TDC molecule oracles when `pytdc` is available. TDC is kept as an optional extra because current PyTDC releases pull a large platform-sensitive ML stack; install it with `uv sync --extra tdc` on a compatible Python if you want TDC SA/QED oracles active. RDKit remains active in the default Docker/HF deployment, and the environment records the active backend in observation metadata.

The default reward mode is `assay_gated`, which gives coarse edit feedback and leaves the strongest quality signal to assays and terminal graders. For early RL warmup, use the curriculum reward mode:

```bash
MOLFORGE_REWARD_MODE=curriculum python inference.py
```

Curriculum mode keeps the official `submission_score` strict, but gives bounded
training reward for useful evidence collection, evidence-supported submit
decisions, and non-submitted near-miss episodes. If the model reaches a strong
evidence package and still fails to submit before the deadline, curriculum mode
adds a small missed-nomination penalty. This prevents small models from seeing
only zero terminal scores while they are still learning when to submit, without
letting endless assay collection become the best behavior. Use this for initial
GRPO curves, then switch back to `assay_gated` for final evaluation.

For curriculum experiments only, you can also restore the older dense edit reward:

```bash
MOLFORGE_REWARD_MODE=dense python inference.py
```

Use randomized training episodes when collecting data or training a policy:

```bash
MOLFORGE_TRAINING_RANDOMIZATION=1 MOLFORGE_RANDOM_SEED=42 python inference.py
```

Keep randomization off for judge-facing baseline runs so scores remain reproducible.

## 2. Recommended Training Plan

Use a two-stage plan:

1. Small SFT warm start
2. RL with verifiable rewards

SFT is only for teaching the model the action schema and basic workflow. RL should do the real environment optimization.

## 3. What SFT Should Teach

Include these example types:

- Valid JSON action formatting
- Correct `acting_role` for each action
- Short `rationale` values that explain the decision without chain-of-thought
- `evidence` lists that cite visible observation facts only
- `expected_effects` dictionaries with directional predictions, not hidden scores
- Specialist message bundles with proposal, approval, objection, assay request, or rejection
- Running cheap/necessary assays before risky submissions
- Editing toward safer fragments when toxicity risk is visible
- Restarting early in the hard sunk-cost scenario
- Submitting only when evidence covers the task constraints
- Handling noisy assay estimates without undoing a high-confidence final candidate at the last moment
- Recovering from low budget by choosing small actions or stopping

Avoid these example types:

- Any example that reads `state.metadata.debug_hidden_properties`
- Any answer that mentions exact hidden objective deltas
- Hidden chain-of-thought or long private reasoning transcripts
- Repetitive message spam just to collect coordination reward
- Premature submit actions without potency/safety evidence
- Examples where missing specialist messages are silently repaired by the runner

## 4. Generate a Starter SFT Dataset

For the first schema warm start, use the strict curriculum dataset. It includes
explicit JSON `null` fields, only the intended top-level action keys, all action
types, all assay tools, all edit subtypes, and valid role/message permissions:

```bash
python scripts/generate_sft_schema_strict_dataset.py \
  --episodes 75 \
  --output data/molforge_sft_schema_strict.jsonl

python scripts/validate_sft_traces.py data/molforge_sft_schema_strict.jsonl
```

Use this file first for Qwen 2B-class SFT:

```text
data/molforge_sft_schema_strict.jsonl
```

The older trace generator is still useful after the model learns the exact
schema, because it provides more policy-like trajectories:

Run:

```bash
python scripts/generate_sft_traces.py --episodes 80 --output data/molforge_sft_traces.jsonl
```

For a more robust dataset:

```bash
python scripts/generate_sft_traces.py \
  --episodes 200 \
  --randomized \
  --output data/molforge_sft_traces_randomized.jsonl
```

The generated records use chat-style JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Before training, spot-check the JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/molforge_sft_traces.jsonl")
for i, line in zip(range(3), path.open()):
    item = json.loads(line)
    print(i, item["metadata"], item["messages"][-1]["content"][:300])
PY
```

## 5. SFT Settings

Start small:

- Dataset size: 200 to 1,000 action examples
- Max sequence length: 2,048 or 4,096
- LoRA rank: 16 or 32
- Learning rate: `1e-4` to `2e-4`
- Epochs: 1 to 3
- Target modules: attention and MLP projection layers
- Save LoRA adapters first; test them before merging

Stop SFT once the model reliably emits valid `MolForgeAction` JSON. Do not overfit it into copying one fixed heuristic path.

## 6. RL Stage

After SFT, run RL/GRPO with MolForge as the verifier environment.

Use these environment settings:

```bash
export MOLFORGE_TRAINING_RANDOMIZATION=1
export MOLFORGE_REWARD_MODE=curriculum
unset MOLFORGE_DEBUG_STATE
```

Once the model starts submitting valid candidates, run a second RL/evaluation
phase with:

```bash
export MOLFORGE_REWARD_MODE=assay_gated
```

Report both curves if possible:

- curriculum reward curve for early learning progress;
- strict terminal `submission_score` before/after for judge-facing task success.

Track these metrics separately:

- Average terminal `submission_score`
- Average terminal `candidate_score`
- Average terminal `budget_score`
- Budget remaining at valid submit
- Invalid action rate
- Policy veto rate
- Budget exhaustion rate
- Repeated assay count
- Loop penalty count
- Coordination score
- Evidence score
- Submitted-without-evidence count
- Constraint margin score
- Number of actions before submit

Inspect generations every few hundred updates. A rising reward is not enough if the model learns to spam messages, submit without evidence, or memorize the three default scenarios.

## 7. Evaluation Protocol

Use three evaluations:

1. Deterministic public tasks
   Run with randomization off and compare to `python inference.py`.
2. Randomized training tasks
   Run with `MOLFORGE_TRAINING_RANDOMIZATION=1`.
3. Holdout tasks
   Add new scenario configs or fragment perturbations not present in SFT traces.

A trained model should improve terminal submission score while keeping invalid actions and evidence-free submissions low.

For the full testing protocol, including how to compare curriculum reward
against strict evaluation, see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## 8. Model Choice

Recommended starting point:

- `unsloth/Qwen3.5-2B` for the lightest serious iteration loop
- `unsloth/Qwen3-4B-Instruct-2507` if you can afford a little more VRAM and want stronger JSON/tool following

Why:

- Qwen3.5 has 0.8B, 2B, and 4B Unsloth fine-tuning support.
- The 2B class should be fast enough for repeated MolForge SFT/RL experiments.
- The 4B class is still lightweight, but should be more reliable for structured action generation.

Use `Qwen3.5-0.8B` only for plumbing tests. It is useful to verify the training loop, but likely too weak to judge the environment.

If you have more GPU budget:

- `unsloth/Qwen3-8B` or a current Qwen3/Qwen3.5 8B-class instruct model

If you specifically want alternate-family baselines:

- `unsloth/Llama-3.1-8B-Instruct`
- Gemma 3/4 small instruct models can be tested, but prefer Qwen first because the current Unsloth Qwen3.5 fine-tuning path is clearer for 2B/4B RL iteration.

For the hackathon, prefer faster iteration over maximum model size. A clean 4B model trained well against this environment is more useful than a larger model that only runs a few noisy experiments.

## 9. Honest Inference Reporting

`inference.py` has no heuristic fallback. It requires a configured model and exits with an error if the model is missing, times out, or emits unparsable action JSON.

`local_inference.py` also has no heuristic policy fallback and does not patch missing team messages into model outputs. If a model omits reviewer communication, that weakness should appear as missing-review penalties and a lower `coordination_score`.

For real model evaluation, run:

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=your-model \
HF_TOKEN=your-token \
python inference.py
```

Use the deterministic trace policy only for SFT data generation, not for reporting model scores.
