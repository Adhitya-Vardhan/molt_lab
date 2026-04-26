---
title: MolForge
emoji: 🧪
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
---

# MolForge

This repository implements an OpenEnv-compatible reinforcement learning environment for **medicinal chemistry lead optimization**. The agent does not directly see the true biological properties of the candidate molecule. Instead, a specialist team iteratively edits a KRAS G12C candidate under limited assay budget, partial observability, and strict safety constraints, receiving a noisy simulated output, and is rewarded for discovering a highly potent, synthesizable, and safe drug candidate.

The environment is designed as a **partially observable Markov decision process (POMDP)** with:
- hidden ground-truth molecular properties and scenario constraints
- hidden target mutation traps
- visible task metadata, team communication, assay results, and remaining budget
- dense step-wise reward (in curriculum mode) plus terminal reward for submission quality

At a high level, each episode looks like this:
1. `reset()` picks a biological scenario (e.g. `level_1_medium`) and seeds the simulator.
2. The agent receives a `MolForgeObservation` describing the task, the starting molecule scaffold, and the current visible state.
3. The agent (acting as different roles) submits a `MolForgeAction` such as `edit`, `run_assay`, `propose_nomination`, or `submit`.
4. The **Governance rule engine** checks whether the action is valid, requiring multi-agent consensus for final decisions.
5. The transition engine updates the molecule, spends the assay budget, and returns oracle readings.
6. The reward computer scores the step based on whether the action was invalid, vetoed, or successful.
7. The environment returns a new observation with updated history, assay readings, and reward.
8. The episode ends when the agent successfully submits the molecule, exhausts its budget, or reaches the maximum step horizon.

---

## Hidden state vs Visible state

### Hidden state
The simulator keeps ground-truth properties that the agent never directly sees. It contains:
- The true underlying scoring functions for `potency`, `safety`, and `synthesizability`.
- Sunk-cost traps and late-stage target mutations (e.g., in `level_2_hard`).
- The strict constraints required for a valid submission.
- The remaining hidden milestones for the scenario.

### Visible state
The agent only sees `MolForgeObservation`, which includes:
- The current `TaskSpec` and `scenario_id`.
- Pipeline history and previous actions.
- The current molecular scaffold (in SMILES format).
- The `budget_used` and `remaining_budget`.
- Responses from the `run_assay` oracle (TDC predictors and RDKit descriptors).
- The `GovernanceStatus` showing which specialist agents have approved or objected.
- The `step_reward_breakdown`.

This separation is what makes the environment a POMDP rather than a fully observed simulator.

---

## Repository files navigation

### `models.py`
Defines the Pydantic contracts that all other modules use:
- `MolForgeAction`: One structured step chosen by the agent. Fields include `action_type`, `acting_role`, `tool_name`, `slot`, `fragment`, and `rationale`.
- `MolForgeObservation`: What the agent can see after each step; includes `current_molecule`, `last_transition_summary`, `reward_breakdown`, and `governance_status`.
- `MolForgeState`: The internal tracked state including `episode_id`, `step_count`, and `invalid_action_count`.

### `server/scenarios.py`
This is where episodes come from. It defines a curated library of three biological scenarios, each bundling a starting scaffold, a budget, and a specific molecular target:
- `level_0_easy`: Potency-first optimization with a generous budget and a starting scaffold that is one or two edits from success.
- `level_1_medium`: Multi-objective optimization with safety as a hard constraint and moderate budget pressure.
- `level_2_hard`: A sunk-cost trap plus late target mutation. The initial scaffold family has a hidden liability, and the best policy is often to restart early.

### `server/actions.py` & `server/governance.py`
The rule engines enforcing scientific and procedural constraints before each action is applied:
- `run_assay`: Costs budget. Evaluates the current molecule using TDC Oracles and RDKit logic.
- `edit`: Replaces a specific R-group slot with a new fragment from the fragment library. Clears previously gathered evidence.
- `submit`: Ends the episode. Triggers the final evaluation grader. 
- **Governance**: Certain actions require multi-agent consensus. If the `Lead Chemist` tries to submit without the `Safety Specialist`'s approval, the action is vetoed.

### `server/molforge_environment.py`
This is the orchestration layer that ties everything together.
On `reset()` it:
- Generates a task scenario.
- Clears the message log, history, and resets the molecule to the default scaffold.

On `step()` it:
- Checks governance rules and validates the action.
- Executes the action (e.g. replacing an R-group fragment or running an assay).
- Computes reward (via Curriculum or Assay-Gated mode).
- Builds the next `MolForgeObservation`.

---

## What actually happens on one step
Here is the concrete order of operations for `env.step(action)`:
1. Increment the step counter.
2. Run validation checks. If the action format is invalid, return a failure report and a `-1.0` reward.
3. Assess **Governance**. If a required specialist agent vetoes the action, the action is blocked and penalized.
4. Execute the action (`edit`, `run_assay`, `submit`).
5. Deduct oracle budget if `run_assay` was called.
6. Compute decomposed reward from the state transition (e.g., getting penalized for redundant assays).
7. If the episode is ending (via `submit`, max steps, or zero budget), compute the terminal `submission_score`.
8. Return an observation that exposes the visible summary but not the hidden truth.

---

## Typical successful pipeline
Most scenarios reward a sensible experiment order similar to:
1. `run_assay` (Assay potency and safety of the baseline molecule).
2. `edit` (Swap an R-group fragment to improve a weak property).
3. `run_assay` (Gather new evidence for the modified molecule).
4. `propose_nomination` (Discuss the findings with the multi-agent review board).
5. `submit` (Finalize the candidate).

The exact best sequence depends on the scenario. In `level_2_hard`, the best strategy is often to `restart` the entire scaffold immediately rather than wasting budget on a doomed trajectory.

---

## Reward Strategy & Episode termination

MolForge uses two distinct reward settings for different purposes:

**1. Training / RL Warmup (`MOLFORGE_REWARD_MODE=curriculum`)**
- Gives partial credit at the end of an episode even if the model didn't submit, provided it gathered useful evidence. 
- It actively prevents "reward hacking" by penalizing assay-spamming, and giving massive multipliers to successful submissions.

**2. Judge-Facing Evaluation (`MOLFORGE_REWARD_MODE=assay_gated`)**
- Strict OpenEnv hackathon rules.
- If the agent does not formally `submit` the candidate, the final score is `0.0`. 
- No partial credit is given for just gathering evidence.

An episode ends when one of the following happens:
- The agent explicitly chooses `submit`.
- Resources (oracle budget) are exhausted.
- The environment reaches `MAX_STEPS`.

---

## Installation & Usage
The package requires Python ≥ 3.10.
```bash
pip install "openenv-core[core]>=0.2.3" pydantic transformers trl peft datasets
```

### 1. In-process environment
Use `MolForgeEnvironment` when you want direct Python access with full structured observations:
```python
from models import MolForgeAction
from server.molforge_environment import MolForgeEnvironment

env = MolForgeEnvironment()
obs = env.reset()

action = MolForgeAction(
    action_type="run_assay",
    acting_role="Lead Chemist",
    tool_name="potency_oracle",
    rationale="Need to gather baseline potency evidence."
)
obs = env.step(action)
print(obs.reward)
print(obs.last_transition_summary)
```

### 2. RL Training Notebook
We have provided a cleanly documented `issue/molforge_grpo_official_submission.ipynb` which demonstrates exactly how to fine-tune a Qwen3.5 model using TRL's GRPO trainer natively against this OpenEnv environment.
