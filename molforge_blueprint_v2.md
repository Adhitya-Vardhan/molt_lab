# MolForge: End-to-End System Blueprint

This document is the implementation blueprint for MolForge. It translates the project idea into a system-level design that can guide environment development, grader design, multi-agent orchestration, training setup, and evaluation.

The goal is to keep the original problem framing intact while making the system robust, measurable, and clearly aligned to the hackathon themes.

---

## 1. Purpose

MolForge is a partially observable scientific design environment where an agent, or a team of agents, iteratively edits molecules under a limited oracle budget to solve high-stakes biomedical challenges. **Currently configured for KRAS G12C (a notoriously difficult, formerly "undruggable" cancer target)**, the agent must navigate a vast chemical space to improve target-specific binding while respecting strict safety and feasibility constraints.

Unlike typical generative AI that acts as a "slot machine" for one-shot molecule generation, MolForge places the agent in an **Epistemic Sandbox**. The model cannot know if an edit is successful until it explicitly spends 'budget' to run an empirical assay, forcing it to execute the true scientific method.

The environment is designed to test:

1. Long-horizon planning under delayed reward
2. Scientific world modeling under partial observability
3. Multi-agent coordination with role-specific incentives
4. Reward-driven improvement through structured feedback

### Core Product Goal

Train and evaluate LLM-based agents on realistic scientific decision-making rather than one-shot molecule generation.

### Primary Output of the System

For each episode, the system should produce:

1. A complete trajectory of states, actions, messages, assay results, and rewards
2. A final candidate molecule
3. A transparent report card showing why the submission succeeded or failed

### Theme Alignment

- Primary: Theme 3.1 Professional Tasks
- Secondary: Theme 1 Multi-Agent Interactions
- Secondary: Theme 2 Long-Horizon Planning and Instruction Following
- Optional extension: Theme 4 Self-Improvement

---

## 2. Design Principles

The following principles govern the implementation:

### 2.1 Environment First

MolForge should behave like a real environment with state transitions, resource constraints, partial observability, and measurable outcomes. It should not feel like prompt engineering around a static scoring script.

### 2.2 Structured Actions Over Free-Form Guessing

Agents should interact with the environment through typed actions and typed messages. This improves learnability, debuggability, and reward reliability.

### 2.3 Reward Transparency

Every reward should be decomposable into named components so that training curves, ablations, and judge-facing demos are easy to explain.

### 2.4 Robustness Against Gaming

The environment must prevent easy exploitation through repetition, verbosity, invalid edits, equivalent-state loops, or reward farming.

### 2.5 Role Separation

If MolForge runs in multi-agent mode, agents must have distinct observations, permissions, and incentives. Otherwise the system will collapse into a set of identical copies of the same agent.

### Theme Alignment

- Theme 3.1: Real dynamic system design
- Theme 1: Role separation and coordinated decision making
- Theme 2: Structured long-horizon planning under budget and step limits

---

## 3. Theme Mapping Matrix

| System Component | Purpose | Primary Theme | Secondary Themes |
| --- | --- | --- | --- |
| Core episode loop | Scientific workflow under constraints | Theme 3.1 | Theme 2 |
| Partial observability | Forces assay planning and belief updates | Theme 3.1 | Theme 2 |
| Oracle budget | Creates long-horizon tradeoffs | Theme 2 | Theme 3.1 |
| Multi-agent specialist roles | Creates coordination and conflict | Theme 1 | Theme 3.1 |
| Structured inter-agent messages | Trainable communication layer | Theme 1 | Theme 2 |
| Grader engine | Enables reward-based learning | Theme 3.1 | Theme 2 |
| Curriculum scenarios | Escalates difficulty over time | Theme 4 | Theme 2 |
| Trace logger and report card | Makes behavior legible and judgeable | Theme 3.1 | Theme 1 |
| Baselines and ablations | Demonstrates actual improvement | Judging criterion support | Theme 1, 2, 3.1 |

---

## 4. System Overview

MolForge is a turn-based environment with three layers:

1. **Environment Core**
   Handles state, turn progression, validation, transitions, termination, and logging.
2. **Oracle Layer**
   Exposes computational tools such as property evaluation, toxicity estimation, docking, and any future external scientific tools.
3. **Agent Layer**
   Contains either a single policy or multiple specialized policies that communicate, propose actions, request assays, and decide when to submit.

### End-to-End Flow

1. Load a scenario
2. Initialize molecule, target, hidden property state, and budget
3. Provide observations to the active agent set
4. Run action selection and message passing
5. Validate and execute the chosen environment action
6. Call graders and update reward
7. Log the full transition
8. Repeat until submission, budget exhaustion, or max-step termination

### Theme Alignment

- Theme 3.1: Full workflow environment
- Theme 1: Agent layer with specialist interaction
- Theme 2: Repeated stateful decision making

---

## 5. Episode Model

### 5.1 Episode Inputs

Each episode is defined by a `ScenarioConfig` object containing:

*(Note: The environment should employ **Domain Randomization** here. Budget, starting scaffold, and internal oracle noise seeds should be randomly perturbed (±20%) each episode to prevent the agent from memorizing specific edit paths.)*

- `scenario_id`
- `task_brief`
- `target_name`
- `starting_scaffold`
- `oracle_budget`
- `max_steps`
- `constraint_set`
- `difficulty_level`
- `enabled_tools`
- `enabled_roles`

### 5.2 Episode Lifecycle

#### Reset

The environment samples or loads a scenario and initializes:

- Current molecule
- Canonical molecular identity
- Remaining budget
- Step counter
- Assay cache
- Message history
- Edit history
- Visited state set
- Hidden property state

#### Interaction Loop

Each turn contains six stages:

1. Observation generation
2. Optional inter-agent messaging
3. Decision and action proposal
4. Validation and transition
5. Reward calculation
6. Logging and termination check

#### Termination Conditions

An episode ends when:

- The agent calls `submit()`
- Budget reaches zero
- Step count reaches `max_steps`
- A hard failure condition is triggered

### 5.3 Output Artifacts

Each completed episode produces:

- Final molecule
- Terminal score
- Reward breakdown
- Constraint satisfaction summary
- Full trajectory trace
- Final report card

### Theme Alignment

- Theme 2: Long-horizon episode structure
- Theme 3.1: Realistic scientific workflow

---

## 6. State Model and Partial Observability

MolForge should be explicitly modeled as a POMDP.

### 6.1 Global Environment State

The full environment state includes:

- Current canonical molecule
- Molecule edit graph and history
- Hidden true and predicted properties
- Oracle budget and per-tool usage
- Scenario constraints
- Warnings and violation flags
- Agent message log
- Submission status

### 6.2 Public State

Information available to all agents:

- Task brief
- Current molecule with atom indices
- Remaining shared budget
- Past executed edits
- Past assay outputs that were explicitly revealed
- Current step number

### 6.3 Hidden State

Information not immediately visible:

- Unqueried property values
- Unused oracle results
- Future consequences of current edits
- Cached internal metrics not yet surfaced to agents

### 6.4 Per-Agent Observations

Each role receives a filtered observation:

#### Lead Chemist

- Molecule structure
- Edit history
- Available edit actions
- Visible assay summaries
- Target objective

#### Toxicologist

- Toxicity assay outputs
- Risk history
- Constraint thresholds
- Safety warnings

#### Assay Planner

- Budget ledger
- Tool costs
- Tool usage history
- Estimated information value of assays

#### Optional Process Chemist

- Synthesis complexity metrics
- Route warnings
- Feasibility flags

### Implementation Requirement

Observations must be emitted as structured objects, not only as concatenated text prompts.

### Theme Alignment

- Theme 3.1: Partial observability and world modeling
- Theme 1: Different views for different roles
- Theme 2: Belief updates over time

---

## 7. Action Space

Actions should be discrete, typed, and validated before execution.

### 7.1 Molecule Edit Actions

- `add_fragment(position, fragment)`
- `substitute(position, fragment)`
- `remove(position)`
- `undo_last_edit()` if enabled for the scenario

### 7.2 Oracle and Tool Actions

- `evaluate_properties()`
- `assay_toxicity()`
- `dock_target(target)`
- `estimate_synthesizability()` if enabled
- `evaluate_novelty()` if enabled
- `search_literature(query)`: Information retrieval action to search patent databases/literature for structural hints.
- `run_md_simulation(molecule)`: High-fidelity, extremely expensive oracle that provides near-perfect ground truth at massive budget cost.

#### Uncertainty-Aware Oracle Returns

Oracles do NOT return single point estimates. They return a **distribution with a confidence interval** that narrows with repeated assays on the same molecule:

```json
{
  "tool": "assay_toxicity",
  "result": {
    "hERG_risk": 0.72,
    "confidence_interval": [0.45, 0.99],
    "n_runs": 1
  }
}
```

Calling the same assay again on an unchanged molecule spends budget but tightens the CI (e.g., `[0.62, 0.82]` after `n_runs: 2`). This forces agents to reason about **epistemic uncertainty**: *"Do I explore a new property, or reduce uncertainty on a critical measurement I already have?"*

### 7.3 Communication Actions

These are used only in multi-agent mode:

- `propose_edit(payload)`
- `request_assay(tool_name, reason)`
- `raise_objection(risk_type, severity, evidence)`
- `approve_action(action_id)`
- `reject_action(action_id, reason)`
- `recommend_submit(summary)`

### 7.4 Governance Actions

- `submit()`
- `defer()`
- `ask_for_revision(agent_id, reason)`
- `restart_from_new_scaffold()`: **Abandon Ship action.** Discards the current molecule and requests a fresh starting scaffold from the scenario pool. Budget already spent is NOT refunded. This tests whether the agent can recognize sunk-cost situations and strategically pivot rather than endlessly polishing a doomed candidate.

### 7.5 Action Cost Ledger

Every action consumes budget and simulated time. Example parameters to ground the environment:

| Action | Cost (Budget) | Time (Simulated Days) |
| --- | --- | --- |
| `evaluate_properties` | $50 | 0.1 |
| `search_literature` | $100 | 0.5 |
| `dock_target` | $300 | 1.0 |
| `assay_toxicity` | $2,000 | 5.0 |
| `run_md_simulation` | $15,000 | 14.0 |

### 7.6 Action Validation Rules

Every action must pass:

1. Syntax validation
2. Role permission validation
3. Chemistry validity checks
4. Budget sufficiency checks
5. Scenario policy checks

If validation fails, the environment returns a structured failure object rather than a generic error string.

Example:

```json
{
  "status": "invalid",
  "error_code": "VALENCE_FULL",
  "message": "Position 14 cannot accept an additional fragment",
  "budget_spent": 0
}
```

### Theme Alignment

- Theme 3.1: Tool-mediated workflow interaction
- Theme 1: Communication and approval mechanics
- Theme 2: Action sequencing and error recovery

---

## 8. Agent Architecture

MolForge should support both single-agent and multi-agent modes.

### 8.1 Single-Agent Mode

A single policy receives the public scientific state and acts directly. This serves as:

- The baseline for ablation
- A simpler initial implementation path
- A way to prove that multi-agent behavior adds value later

### 8.2 Multi-Agent Mode

Recommended initial architecture:

1. **Lead Chemist**
   Owns molecular edit proposals
2. **Toxicologist**
   Reviews risk and flags unsafe directions
3. **Assay Planner**
   Decides whether spending budget on an oracle call is justified

Optional fourth role:

4. **Process Chemist**
   Evaluates synthetic complexity and practical feasibility

### 8.3 Role Responsibilities

| Role | Main Responsibility | Can Edit Molecule | Can Approve Assays | Can Veto Submission |
| --- | --- | --- | --- | --- |
| Lead Chemist | Generate and revise edit proposals | Yes | No | No |
| Toxicologist | Flag risk and safety regressions | No | Request only | Yes on hard safety constraints |
| Assay Planner | Allocate oracle budget | No | Yes | Yes on budget grounds |
| Process Chemist | Flag feasibility concerns | No | Request only | Optional |

### 8.4 Role Incentives

Each role should optimize a slightly different local objective:

- Lead Chemist: potency and progress
- Toxicologist: safety and avoidance of severe violations
- Assay Planner: information gain per unit budget
- Process Chemist: feasibility and complexity control

### 8.5 Role Collapse Prevention

To avoid all agents behaving like the same model:

1. Use different observation slices
2. Use different action permissions
3. Use different local reward signals
4. Track agent-specific contribution metrics

### Theme Alignment

- Theme 1: Multi-agent interactions
- Theme 3.1: Professional specialist workflow
- Theme 2: Coordination over multiple turns

---

## 9. Inter-Agent Communication Protocol

Multi-agent interaction must be explicit, bounded, and inspectable.

### 9.1 Communication Design Goals

- Make coordination visible in logs and demos
- Prevent unbounded discussion loops
- Allow disagreement with structured resolution

### 9.2 Message Schema

All messages should follow a typed schema:

```json
{
  "message_id": "msg_014",
  "sender": "toxicologist",
  "receiver": "lead_chemist",
  "type": "raise_objection",
  "reference_action_id": "act_009",
  "payload": {
    "risk_type": "hERG_risk",
    "severity": "high",
    "evidence": "Predicted risk increased from 0.31 to 0.78"
  }
}
```

### 9.3 Allowed Message Types

- `proposal`
- `objection`
- `risk_flag`
- `assay_request`
- `approval`
- `rejection`
- `revision_request`
- `submission_recommendation`

### 9.4 Communication Limits

To avoid message farming:

- Limit messages per turn
- Apply a small cost for unnecessary discussion
- Collapse duplicate objections into a single logged event

### 9.5 Conflict Resolution Policy

Recommended default policy:

1. Lead Chemist proposes
2. Toxicologist and Assay Planner review
3. If no hard objection exists, action proceeds
4. If objections exist, the Lead Chemist must revise or appeal
5. Hard constraint vetoes override the proposal unless explicitly relaxed by scenario config

### Theme Alignment

- Theme 1: Real coordination, conflict, and negotiation
- Theme 2: Multi-step communication under constrained horizons

---

## 10. System Policies

Policies are environment-level rules that determine what is allowed, how conflicts are resolved, and how the system handles edge cases.

### 10.1 Environment Policies

- Hard chemistry validity policy
- Hard budget policy
- Scenario-specific constraint policy
- Termination policy
- Logging policy

### 10.2 Agent Governance Policies

- Who can propose edits
- Who can authorize expensive tools
- Which roles can raise hard vetoes
- How revisions are requested and counted

### 10.3 Anti-Gaming Policies

- Penalize repeated no-op or equivalent actions
- Penalize repeated assay calls with no state change
- Penalize oscillation between previously visited states
- Canonicalize molecules to prevent representation exploits
- Cap discussion rounds per decision

### 10.4 Reasoning Policy

Reasoning should not be rewarded purely for length. Instead, reward should depend on whether the explanation is consistent with the chosen action and the observed outcome.

Example:

- Positive if the agent claims a polarity-based safety improvement and the resulting property change supports that claim
- Negative if the explanation is chemically inconsistent with the observed results

### Theme Alignment

- Theme 2: Recovery from poor actions and policy-constrained planning
- Theme 3.1: Reliable system behavior in a dynamic tool environment

---

## 11. Grader Architecture

The grader system should be modular and transparent. Each grader computes a named score and emits both scalar values and structured evidence.

### 11.1 Grader Engine

The grader engine runs after every transition and on terminal submission.

Inputs:

- Previous state
- Current state
- Executed action
- Relevant messages
- Oracle outputs
- Scenario constraints

Outputs:

- Step reward components
- Penalty components
- Terminal reward components if applicable
- Human-readable explanation

### 11.2 Step Graders

#### Validity Grader

Checks whether the action was legal and chemically valid.

Outputs:

- `validity_score`
- `error_code`
- `failure_reason`

#### Progress Grader

Measures movement toward scenario objectives.

Possible signals:

- Change in binding score
- Change in toxicity score
- Change in QED
- Change in SA score

#### Efficiency Grader

Rewards useful information gathering and penalizes wasteful oracle spending.

Signals:

- Assay value relative to gained information
- Unnecessary repeated tool usage
- Budget usage efficiency

#### Coordination Grader

Used in multi-agent mode.

Rewards:

- Correctly flagged risky proposals
- Productive revisions
- Final agreement with fewer wasted turns

Penalizes:

- Ignored valid objections
- Unnecessary back-and-forth
- Role violations

#### Reasoning Consistency Grader

Measures whether the explanation matches the action and the observed outcome.

This grader should replace any naive token-count based bonus.

#### Novelty Grader

Optional but useful for preventing trivial submissions.

Signals:

- Distance from starting scaffold
- Distance from previously visited states
- Redundancy penalties

### 11.3 Terminal Graders

#### Submission Quality Grader

Evaluates the final molecule against all scenario objectives.

#### Constraint Satisfaction Grader

Checks hard and soft constraints separately.

#### Budget Outcome Grader

Rewards solving the task without unnecessary oracle spending.

#### Report Card Generator

Produces the final explanation visible to users and judges.

### 11.4 Reward Aggregation

**CRITICAL HACKATHON STRATEGY**: To ensure stable training in a short timeframe, avoid arbitrary linear combinations of unbounded weights. Adopt an RLVR (Verifiable Reward) approach with bounded components.

Recommended structure:

```text
# Dense shaping rewards
R_step = (validity_flag * 1.0) - (0.5 * num_soft_violations)

# Sparse terminal reward (The Ground Truth)
R_terminal =
  (100.0 if meets_hard_constraints and beats_baseline else 0.0) +
  (budget_efficiency_bonus) # e.g. max(0, 1 - budget_used/total_budget)
```

By keeping step rewards minimal and focusing on a massive sparse terminal reward for verifiable success, the training curves will reflect actual scientific progress rather than reward farming.

### Theme Alignment

- Theme 3.1: Reward from real workflow outcomes
- Theme 2: Delayed and decomposed reward
- Theme 1: Coordination-aware grading in multi-agent mode

---

## 12. Core Workflow: Single-Agent Mode

This workflow is the simplest executable version of MolForge.

### Single-Agent Decision Loop

1. Receive observation
2. Choose one action
3. Environment validates it
4. Oracle layer runs if needed
5. Graders compute reward
6. State updates
7. Repeat until termination

### Why This Mode Matters

- Fastest path to a working environment
- Baseline for comparison
- Useful for debugging reward logic before multi-agent complexity is added

### Theme Alignment

- Theme 3.1: Scientific workflow simulation
- Theme 2: Long-horizon sequential decision making

---

## 13. Core Workflow: Multi-Agent Mode

This workflow is the recommended target design once the single-agent environment is stable.

### Multi-Agent Turn Protocol

1. Environment emits role-specific observations
2. Lead Chemist proposes an edit or tool action
3. Toxicologist reviews and raises objections if needed
4. Assay Planner approves or rejects budget-heavy actions
5. Optional Process Chemist comments on feasibility
6. Lead Chemist revises, defends, or withdraws the proposal
7. Final executable action is selected
8. Environment applies the action
9. Grader engine computes both outcome reward and coordination reward

### Multi-Agent Success Condition

The system should outperform the single-agent baseline on at least one of:

- final reward
- constraint satisfaction rate
- oracle efficiency
- bad-edit avoidance

### Theme Alignment

- Theme 1: Multi-actor interaction
- Theme 3.1: Realistic team-based scientific workflow
- Theme 2: Long-horizon coordination and recovery

---

## 14. Oracle Layer Design

The oracle layer abstracts all expensive or scientific computations.

### 14.1 Oracle Types

- Property oracle
- Toxicity oracle
- Docking oracle
- Feasibility oracle
- Optional novelty or diversity oracle

### 14.2 Oracle Wrapper Requirements

Each oracle wrapper must:

1. Expose a stable interface
2. Support deterministic seeding where possible
3. Cache repeated calls on identical canonical molecules
4. Log latency, cost, and outputs
5. Return structured outputs

Example response:

```json
{
  "tool": "assay_toxicity",
  "status": "ok",
  "molecule_id": "mol_023",
  "result": {
    "hERG_risk": 0.28,
    "hepatotox_risk": 0.12
  },
  "budget_spent": 3
}
```

### 14.3 Oracle Reliability Rules

- Never recompute if the canonical molecule and tool call are unchanged unless explicitly requested
- Use cache hits to prevent score noise and budget inconsistencies
- Mark unstable oracle outputs in logs if variance is known

### Theme Alignment

- Theme 3.1: Tool-integrated professional workflow
- Theme 2: Planning under expensive information acquisition

---

## 15. Scenario and Curriculum Design

Scenarios define what the agent is trying to optimize and how hard the episode is.

### 15.1 Scenario Families

- Potency improvement
- Potency plus toxicity constraint
- Multi-objective optimization with hard thresholds
- Mutation adaptation and recovery

### 15.2 Difficulty Tiers

#### Level 1

Simple property improvement with generous budget and minimal conflict

#### Level 2

Multi-objective optimization with moderate budget pressure

#### Level 3

Adversarial constraints, tighter budgets, higher ambiguity, and more role conflict

### 15.3 Curriculum Scheduling

Training should begin with simpler scenarios and gradually introduce:

- lower budgets
- more hidden information
- more conflicting constraints
- more agent roles

### 15.4 Theme 4 Extension

MolForge can support self-improvement by allowing the environment to:

- generate harder scenario variants
- mutate constraint sets
- escalate budget pressure
- introduce new target mutations

This is optional and should be treated as an advanced extension, not an MVP requirement.

### Theme Alignment

- Theme 2: Increasing long-horizon difficulty
- Theme 4: Curriculum and adaptive challenge generation
- Theme 3.1: Broader professional task coverage

---

## 16. Innovative Differentiators

These features are designed to push MolForge beyond a well-engineered environment into a genuinely novel RL research contribution. They target the **Environment Innovation (40%)** judging criterion.

### 16.1 Adversarial Target Mutation (Non-Stationary Environment)

Real cancer targets mutate to develop drug resistance. MolForge models this.

**Mechanism:** At a scenario-configured step (or stochastically), the environment silently swaps or mutates the binding target. The agent's next `dock_target()` call returns drastically worse affinity scores. The agent receives no explicit notification — it must **detect the shift from observation alone**, diagnose what changed, and re-strategize its molecular edits.

**Implementation:**
- Store a list of 2-3 related target conformations per scenario (e.g., KRAS G12C wild-type, KRAS G12C/A59T resistance mutant)
- At the mutation step, swap the active target conformation used by the docking oracle
- The agent's observation does NOT reveal the swap; it only sees that docking scores changed

**Why this wins:**
- No other hackathon submission will have a **non-stationary, adversarial environment**
- Directly models the real clinical crisis of drug resistance
- Forces genuine long-horizon adaptation (Theme 2) and world-model updating (Theme 3.1)
- Enables a dramatic pitch: *"MolForge doesn't just test if an agent can find a drug. It tests if the agent can survive when the disease fights back."*

### 16.2 Sunk-Cost Trap Scenarios (The "Abandon Ship" Test)

Some scenarios are intentionally designed so that the starting scaffold has a **fundamental structural flaw** (e.g., a core ring system that always triggers PAINS toxicity filters, or a scaffold that is physically impossible to dock regardless of edits).

**Mechanism:** The only way to achieve a positive terminal reward is to call `restart_from_new_scaffold()` early, accept the sunk budget loss, and work with a fresh starting molecule using the remaining budget.

**Why this wins:**
- Current LLMs are notoriously bad at recognizing sunk-cost fallacy — they will endlessly polish a doomed molecule
- This is a genuine **cognitive bias test** that no other RL environment probes
- It creates powerful demo moments: *"Watch how the untrained model wastes 80% of budget on a dead-end, while the trained model learns to cut its losses at step 3."*
- Trivially easy to implement (1 new action + 2 trap scenarios)

### 16.3 Uncertainty-Aware Oracles (Epistemic Reasoning)

As defined in Section 7.2, oracles return distributions rather than point estimates. Early assays are noisy and uncertain; repeated assays on the same molecule tighten confidence intervals.

**Why this wins:**
- Forces genuine **epistemic reasoning**: the Assay Planner must decide between exploring new properties vs. reducing uncertainty on existing measurements
- Models real laboratory science where initial experimental results are always noisy
- Creates a natural tradeoff between breadth (many cheap noisy assays) and depth (fewer expensive high-confidence assays)
- Enables the Active Learning narrative: *"Our agent learns to allocate budget like a real scientist — investing in certainty where it matters most."*

### Theme Alignment

- Theme 3.1: Non-stationary professional environments, epistemic reasoning, real-world failure modes
- Theme 2: Long-horizon adaptation to environment shifts, strategic pivoting
- Theme 4: Adversarial curriculum that escalates unpredictably
- Theme 1: Critic agents must detect and communicate environment shifts to the Proposer

---

## 16. Robustness Requirements

This section defines what must be true before the system is considered implementation-ready.

### 16.1 Chemistry Robustness

- Canonicalize SMILES after every successful edit
- Reject invalid chemistry before reward calculation
- Track equivalence classes of molecules to avoid duplicate-state exploits

### 16.2 Reward Robustness

- Normalize reward components
- Clip extreme deltas
- Keep step and terminal rewards separate
- Emit a full reward breakdown for every transition

### 16.3 Multi-Agent Robustness

- Enforce role permissions strictly
- Bound message rounds
- Log message provenance and impact
- Penalize non-productive coordination loops

### 16.4 Reproducibility

- Scenario seeds must be fixed for evaluation
- Oracle outputs should be deterministic where possible
- Episode traces must be replayable from logs

### 16.5 Failure Handling

All failures should be categorized:

- parse failure
- permission failure
- chemistry failure
- budget failure
- oracle failure
- policy veto

### Theme Alignment

- Theme 3.1: Professional-grade environment reliability
- Theme 1: Stable agent interaction design
- Theme 2: Robust long-run behavior

---

## 17. Logging, Analytics, and Report Cards

The environment must be easy to inspect during training and demo.

### 17.1 Transition Log

Every step should log:

- state id
- action id
- acting role
- observation hash or summary
- messages exchanged
- oracle calls
- reward breakdown
- budget change
- termination flag

### 17.2 Episode Summary

Each episode summary should include:

- scenario id
- final molecule
- total reward
- terminal reward components
- number of edits
- number of oracle calls
- number of objections raised
- number of invalid actions

### 17.3 Final Report Card

The report card shown at `submit()` should summarize:

1. What improved
2. What constraints were met
3. Which tradeoffs were accepted
4. Whether the final candidate beat the baseline scaffold

### Theme Alignment

- Theme 3.1: Explainable workflow traces
- Theme 1: Legible multi-agent behavior
- Supports judging criteria on storytelling and improvement

---

## 18. Evaluation Plan

The blueprint should include evaluation from day one.

### 18.1 Baselines

Required baselines:

1. Random action baseline
2. Heuristic scripted baseline
3. Single-agent LLM baseline
4. Multi-agent LLM system

### 18.2 Metrics

Core metrics:

- average terminal reward
- success rate on hard constraints
- average oracle budget consumed
- invalid action rate
- submission quality rate
- improvement over starting scaffold

Multi-agent metrics:

- objections raised
- objections that prevented a bad outcome
- revision success rate
- coordination cost per successful episode

### 18.3 Ablations

Recommended ablations:

- with vs without multi-agent mode
- with vs without partial observability
- with vs without reasoning consistency grader
- with vs without anti-loop penalties

### 18.4 Demo-Ready Evidence

For judging, the system should be able to show:

1. A reward curve or success-rate curve
2. A before-and-after trajectory comparison
3. A readable episode trace
4. At least one example where coordination improved the result

### Theme Alignment

- Supports judging criteria directly
- Theme 1: Proves multi-agent value
- Theme 2: Proves long-horizon learning
- Theme 3.1: Proves environment realism

---

## 19. Suggested Implementation Architecture

The following module layout is recommended.

### 19.1 Core Modules

| Module | Responsibility |
| --- | --- |
| `scenario_loader` | Load and validate scenario configs |
| `env_core` | Reset, step, termination, and transition handling |
| `state_manager` | Canonical state object and serialization |
| `action_parser` | Parse and validate agent actions |
| `oracle_manager` | Tool wrappers, caching, and structured responses |
| `grader_engine` | Step and terminal grading |
| `agent_controller` | Role routing and multi-agent scheduling |
| `policy_manager` | Permissions, veto logic, anti-gaming rules |
| `trace_logger` | Step logs, summaries, and replay data |
| `report_card` | User-facing final summary |

### 19.2 Interface Contracts

Each module should operate on typed objects, not raw strings, wherever possible.

Recommended shared objects:

- `ScenarioConfig`
- `EnvState`
- `AgentObservation`
- `AgentAction`
- `ValidationResult`
- `OracleResult`
- `RewardBreakdown`
- `EpisodeSummary`

### 19.3 Development Sequence

Suggested build order:

1. Single-agent environment core
2. Action parser and validation
3. Oracle manager with caching
4. Step graders and terminal graders
5. Trace logger and report card
6. Multi-agent observation routing
7. Multi-agent communication protocol
8. Coordination grader
9. Baselines and ablations

### Theme Alignment

- Theme 3.1: Real environment and tools
- Theme 1: Multi-agent controller and communication stack
- Theme 2: Full long-horizon pipeline

---

## 20. MVP vs Phase 2 Scope

### MVP

The smallest version that should be implemented first:

- Single-agent mode
- Molecule edit actions
- Property and toxicity oracles
- Oracle budget
- Step and terminal graders
- Full trajectory logging
- Fixed evaluation scenarios

### Phase 2

The highest-value extension after the MVP is stable:

- Three-agent specialist workflow
- Structured objections and approvals
- Coordination grading
- Better curriculum scheduling
- More scenario families

### Phase 3

Optional stretch work:

- Process Chemist role
- Adaptive scenario generation
- Self-improvement curriculum
- Additional feasibility or novelty tools

### Theme Alignment

- MVP: Theme 3.1 and Theme 2
- Phase 2: Theme 1
- Phase 3: Theme 4

---

## 21. Non-Goals

To keep the project focused, the following are not MVP requirements:

- Fully realistic medicinal chemistry simulation
- Guaranteed scientific correctness beyond the chosen oracle stack
- Large-scale autonomous literature review
- Production deployment infrastructure

This is a hackathon-grade training environment, not a complete drug discovery platform.

---

## 22. Training Convergence Strategy

This section defines the concrete plan for demonstrating that an LLM agent actually improves through RL training in MolForge. This directly targets the **Showing Improvement in Rewards (20%)** and **Training Script/Pipeline (10%)** judging criteria.

### 22.1 Model Selection

**Recommended models (in order of preference):**

| Model | Parameters | Why |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | 0.8B | Fastest iteration, proven to work in the winning hackathon project |
| `Qwen/Qwen3.5-4B` | 4B | Best quality-to-speed ratio for H100 |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Strong alternative if Qwen has issues |

**Critical:** Start with the **0.8B model first**. Get the reward curve working. Only scale up if time permits. A beautiful reward curve on a 0.8B model beats a crashed training run on a 4B model.

### 22.2 Training Method: GRPO (Group Relative Policy Optimization)

Use **TRL's GRPOTrainer** or **Unsloth's quantized GRPO** (identical to the winning project's approach).

**How GRPO works with MolForge:**
1. Collect N prompts (each prompt = one MolForge observation/state)
2. For each prompt, generate K completions (each completion = a proposed action JSON)
3. Execute each action in the MolForge environment to get the reward
4. GRPO uses the relative ranking of rewards within each group to update the policy

### 22.3 Ensuring Convergence

The single biggest risk is that the model fails to learn. Here is the strategy to guarantee visible improvement:

#### Step 1: Constrain the Action Space (Critical)
Do NOT let the LLM output free-form text. Force it to output **structured JSON** matching a strict schema:

```json
{
  "action": "substitute",
  "position": 4,
  "fragment": "F",
  "reasoning": "Replace methyl with fluorine to reduce lipophilicity"
}
```

If the output fails JSON parsing → immediate reward of -1.0. This gives a massive, trivially learnable signal in the first few training steps.

#### Step 2: Start with a Trivially Easy Scenario
Create a "Level 0" scenario where:
- The starting molecule is 1 edit away from a perfect score
- Budget is enormous (100 units)
- Only 2 possible actions matter
- The correct action is obvious from the observation

This guarantees the reward curve goes up in the first 50 steps. You can show this to judges as proof of learning.

#### Step 3: Curriculum Escalation
Once Level 0 converges, switch to Level 1 (real scenarios). The model should transfer its learned JSON formatting and basic chemistry intuition.

### 22.4 Reward Design for Convergence

Use the sparse RLVR approach from Section 11.4:

```text
R = -1.0    if action fails JSON parsing or chemistry validation
R =  0.0    for valid actions that don't improve the molecule
R = +1.0    for valid actions that improve at least one property
R = +10.0   terminal bonus if final molecule beats starting scaffold
R = -5.0    terminal penalty if final molecule is worse than start
```

This 5-level reward is simple enough for a small model to learn quickly.

### 22.5 Expected Training Timeline (on H100)

| Phase | Duration | Expected Result |
| --- | --- | --- |
| Data collection (rollouts) | 30 min | 500-1000 prompt-action pairs |
| GRPO training (0.8B, 100 steps) | 1-2 hrs | Reward curve visibly increasing |
| GRPO training (4B, 50 steps) | 2-4 hrs | Stronger convergence |
| Evaluation + plots | 30 min | Beautiful graphs for the pitch |

### 22.6 Demo Artifacts to Generate

The training script MUST automatically save these files:

1. `training_reward_curve.png` — The money shot. Reward going up over training steps.
2. `training_loss.png` — Policy loss decreasing.
3. `before_after_trajectory.json` — Side-by-side comparison of an untrained vs. trained agent's episode.
4. `sunk_cost_demo.json` — A trajectory showing the trained agent correctly abandoning a trap scaffold (killer demo).

### 22.7 Training Script Skeleton

```python
from trl import GRPOTrainer, GRPOConfig
from molforge.env import MolForgeEnvironment

def reward_fn(completions, prompts):
    """Execute each completion as an action in MolForge and return rewards."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        env = MolForgeEnvironment(scenario="level_0")
        state = env.restore_from_prompt(prompt)
        try:
            action = json.loads(completion)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        except (json.JSONDecodeError, KeyError):
            rewards.append(-1.0)  # Invalid JSON = harsh penalty
    return rewards

config = GRPOConfig(
    num_generations=4,
    max_completion_length=160,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=prompt_dataset,
)

trainer.train()
```

### Theme Alignment

- Directly supports judging criteria: Showing Improvement in Rewards (20%) and Pipeline Setup (10%)
- Demonstrates that the environment is not just a design document but a working training loop

---

## 23. Final Blueprint Summary

MolForge should be implemented as a stateful scientific environment with explicit actions, modular graders, robust oracle integration, and optional multi-agent coordination. The system must prioritize structured transitions, transparent rewards, role-specific observability, and reproducible evaluation.

If implemented according to this blueprint, MolForge will have:

1. A strong Theme 3.1 foundation through realistic scientific workflow simulation
2. A credible Theme 1 extension through explicit specialist coordination
3. A natural Theme 2 fit through budgeted long-horizon planning
4. A possible Theme 4 extension through adversarial target mutation and curriculum growth
5. **Three genuinely novel features** that no competitor will have: adversarial target mutation, sunk-cost trap scenarios, and uncertainty-aware oracles
6. **A working GRPO training pipeline** with demonstrated reward convergence

The highest-priority implementation path is:

1. Build a stable single-agent environment with Level 0 scenario
2. Get the GRPO training loop running and producing reward curves
3. Add the innovative differentiators (target mutation, abandon ship, uncertainty oracles)
4. Add multi-agent coordination only if time permits

That sequence gives the team the highest probability of having a **working demo with real training results** during judging.
