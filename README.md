---
title: MolForge
emoji: 🧪
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
---

# MolForge

MolForge is an OpenEnv environment for medicinal chemistry lead optimization. A specialist team iteratively edits a KRAS G12C candidate under limited assay budget, partial observability, and strict safety constraints. The environment uses RDKit-backed molecular descriptors and TDC molecule oracles when available, blended with target-specific KRAS surrogate logic so it stays lightweight enough for the OpenEnv hackathon validator while still modeling a real scientific workflow.

## Why this is a real-world task

Medicinal chemists routinely make scaffold edits, request assays, triage toxicity risk, and decide when to stop spending budget on a candidate. MolForge turns that workflow into a stateful environment with constrained actions, hidden property values, specialist review messages, governance/veto rules, grader-backed submissions, and reproducible episode traces.

## Task Set

MolForge rotates through three built-in tasks on successive `reset()` calls:

1. `level_0_easy`
   Potency-first optimization with a generous budget and a starting scaffold that is one or two edits from success.
2. `level_1_medium`
   Multi-objective optimization with safety as a hard constraint and moderate budget pressure.
3. `level_2_hard`
   A sunk-cost trap plus late target mutation. The initial scaffold family has a hidden liability, and the best policy is often to restart early.

Each task has a deterministic grader that outputs scores in the `0.0` to `1.0` range:

- `final_score`
- `potency_score`
- `safety_score`
- `synth_score`
- `novelty_score`
- `candidate_score`
- `constraint_score`
- `budget_score`
- `coordination_score`
- `evidence_score`
- `submission_score`

## Action Space

`MolForgeAction` is a typed Pydantic model with these high-level actions:

- `edit`
  Requires `acting_role="lead_chemist"`, `edit_type`, `slot`, and usually `fragment`
- `run_assay`
  Requires `acting_role="assay_planner"` and `tool_name`
- `submit`
- `restart`
- `defer`

Each action also carries a typed `messages` bundle containing specialist communication such as:

- `proposal`
- `approval`
- `objection`
- `risk_flag`
- `assay_request`
- `rejection`
- `submission_recommendation`

Actions also include bounded, auditable reasoning fields:

- `rationale`: short public explanation
- `evidence`: visible observation facts supporting the decision
- `expected_effects`: directional predictions such as `toxicity="down"` or `budget="down"`

Enabled roles:

- `lead_chemist`
- `toxicologist`
- `assay_planner`
- `process_chemist`

Editable slots:

- `warhead`
- `hinge`
- `solvent_tail`
- `back_pocket`

Available tools:

- `evaluate_properties`
- `dock_target`
- `assay_toxicity`
- `estimate_synthesizability`
- `evaluate_novelty`
- `search_literature`
- `run_md_simulation`

## Observation Space

`MolForgeObservation` includes:

- Current scenario metadata and task brief
- Canonical molecule string plus per-slot fragment assignments
- Remaining budget and step counts
- Visible assay results with confidence intervals
- Role-specific structured observation slices for all specialists
- Structured message log
- Governance state showing approvals, objections, and vetoes
- Constraint status based only on visible evidence
- Reward breakdown for the last transition
- Report card on termination

The hidden state contains the true molecular properties, target-shift logic, whether the current scaffold is a trap series, and the full internal trace used by graders.

The observation metadata also includes the RDKit/TDC surrogate SMILES and active oracle backend flags, for example `{"rdkit": true, "tdc": true}` when both engines are available. RDKit is part of the default deploy path. TDC is an optional extra because current PyTDC releases pull a large platform-sensitive ML stack; install with `pip install -e '.[tdc]'` or `uv sync --extra tdc` on a compatible Python if you want TDC SA/QED oracles active.

## Reward Design

The reward function mixes coarse shaping and sparse terminal bonuses:

- Coarse edit feedback that avoids exposing exact hidden objective deltas
- Information-gain reward for useful assays
- Coordination reward for correct specialist reviews, proposal discipline, and required review coverage
- Penalties for invalid actions, repeated states, wasteful assay repetition, missing reviews, and budget burn
- Evidence penalties for submitting without current potency, toxicity, and synthesis support
- Proportional constraint penalties so worse toxicity, potency, or synthesis violations hurt more than near misses
- Large terminal reward for submitting a molecule that beats baseline while satisfying hard constraints and evidence requirements
- Small budget-efficiency credit for valid evidence-backed submissions, so finishing with unused budget is better than reaching the same candidate wastefully

`final_score` is the single headline scalar for RL/evaluation. It equals strict `submission_score` after a valid formal submission, and gives only small capped partial credit to non-submitted episodes. `candidate_score` and `progress_score` are diagnostic breakdowns only. `progress_score` is deliberately capped by failed constraints, repeated assays, policy-veto loops, and hard-scenario trap failures, so evidence collection alone cannot look like success. `submission_score` remains a formal task-completion score and is `0.0` without a `submit` action.

For early RL, `MOLFORGE_REWARD_MODE=curriculum` adds bounded warmup reward for useful evidence collection, evidence-supported submit decisions, and non-submitted near-miss episodes. It also adds a small missed-nomination penalty when a strong evidence package is ready but the agent lets the deadline pass without submitting. This makes reward curves less sparse for small models, while leaving the official terminal `submission_score` unchanged. Final judge-facing evaluation should still report strict `submission_score` in the default `assay_gated` mode.

This keeps the environment informative for training while making exploitation visible in the logs.

For SFT/RL training guidance, trace generation, randomization flags, and reward-hacking precautions, see [TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md).

## Multi-Agent Workflow

Each environment step represents one coordinated team decision:

1. The acting specialist proposes an executable action.
2. Other required reviewers send typed messages.
3. Governance checks permissions, review coverage, and hard-veto conditions.
4. The environment either executes the action or blocks it with a policy veto.

The specialists do not call each other through separate API turns. A single `MolForgeAction` contains the executable team action plus a typed message bundle. The environment grades that bundle by checking that the acting role made a proposal, required non-acting reviewers responded with the expected message type, and no role used a message it is not allowed to send. Missing reviewers lower both transition reward and terminal `coordination_score`.

Hard-veto behavior currently covers:

- Toxicologist safety vetoes
- Assay Planner budget/evidence vetoes
- Process Chemist feasibility vetoes on hard synth constraints

## Project Layout

- Core models: [models.py](models.py)
  Typed action, observation, state, assay, and governance schemas.
- Scenario definitions: [scenarios.py](scenarios.py)
  Fragment libraries, budgets, task configs, molecule scoring, and deterministic graders.
- Client wrapper: [client.py](client.py)
  Thin OpenEnv client for interacting with the environment.
- Environment entrypoint: [server/molforge_environment.py](server/molforge_environment.py)
  Small orchestration class that wires the focused server mixins together.
- Shared environment utilities: [server/shared.py](server/shared.py)
  Reusable constants, state helpers, assay merging, score normalization, and trace utilities.
- Action execution: [server/actions.py](server/actions.py)
  Edit, assay, submit, restart, and transition-reward logic.
- Governance logic: [server/governance.py](server/governance.py)
  Action validation, reviewer expectations, vetoes, and visible reasoning-grounding checks.
- Observation and grading: [server/views.py](server/views.py)
  Observation assembly, role-specific views, report cards, and terminal graders.
- OpenEnv app: [server/app.py](server/app.py)
  FastAPI/OpenEnv server bootstrap.
- Shared inference helpers: [inference_common.py](inference_common.py)
  Common prompts, JSON extraction, prompt payload shaping, and heuristic team policy.
- Judge runner: [inference.py](inference.py)
  Official OpenAI-client baseline used for validator-style execution.
- Local runner: [local_inference.py](local_inference.py)
  Ollama-native development script that fails loudly when the local model cannot emit a valid action.
- Submission manifests: [openenv.yaml](openenv.yaml), [Dockerfile](Dockerfile), [server/Dockerfile](server/Dockerfile)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
openenv validate
```

## Baseline Inference

The required judge-facing baseline script lives at [inference.py](inference.py) and uses the OpenAI Python client with these environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

If those variables are unavailable or the model request fails, the script exits with an error. There is no heuristic fallback in the judge-facing runner.

Run it with:

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=your-model \
HF_TOKEN=your-token \
python inference.py
```

## Local Testing

For local Ollama testing, use [local_inference.py](local_inference.py) instead of `inference.py`.

This local script:

- talks to Ollama's native `/api/chat` endpoint
- supports `think: false` for reasoning models
- fails loudly when the local model cannot produce parseable action JSON
- keeps the judge-facing script clean and spec-aligned

Run it with:

```bash
OLLAMA_BASE_URL=http://localhost:11434 \
LOCAL_MODEL_NAME=gemma4:e2b \
OLLAMA_THINK=false \
python local_inference.py
```

Reference offline smoke-test baseline from the deterministic trace policy used only for SFT data generation. This is not a model score and should not be reported as frontier-model performance:

- `level_0_easy`: `0.8703`
- `level_1_medium`: `0.8733`
- `level_2_hard`: `0.8883`
- Average final/submission score: `0.8773`

## Deployment

The environment is packaged as a FastAPI OpenEnv Space using the repo-root [Dockerfile](Dockerfile) for validator compatibility, with [server/Dockerfile](server/Dockerfile) retained for OpenEnv-style server packaging. The root manifest is [openenv.yaml](openenv.yaml), so `openenv push` or a Docker Hugging Face Space can deploy it.

## Pre-RL SFT Adapter

The current Qwen3.5 2B QLoRA/SFT v4 adapter is stored in the Hugging Face Space at `adapters/qwen3_5_2b_lora_adapters_compact_v4/`. It is intentionally excluded from the Docker build context so the OpenEnv server stays lightweight, but the adapter files remain available from the Space repository for model debugging.

Before RL, that adapter produced these strict MolForge `final_score` values:

- `level_0_easy`: `0.1167`
- `level_1_medium`: `0.1167`
- `level_2_hard`: `0.0800`
