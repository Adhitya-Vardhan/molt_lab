# MolForge Real-World Workflow Mapping

MolForge should feel like a compressed medicinal-chemistry lead-optimization
program, not a one-shot molecule generator.

The real-world pattern is:

1. A team starts with a scaffold.
2. Chemists propose edits based on structure-activity reasoning.
3. Assay teams spend limited budget to measure uncertain properties.
4. Safety and process specialists veto risky or impractical candidates.
5. The team decides whether to keep optimizing, restart, or nominate a lead.
6. Success depends on evidence, not only on the final molecule.

This is exactly the shape MolForge should copy.

## Real-World Loop

### 1. Design Hypothesis

Real teams do not mutate molecules randomly. A medicinal chemist proposes a
change with an intended purpose:

- improve potency;
- reduce toxicity;
- improve solubility or ADME;
- simplify synthesis;
- escape a known scaffold liability.

MolForge equivalent:

- `edit`
- `rationale`
- `expected_effects`
- `evidence`

The model should not only choose a fragment. It should say what scientific
pressure that edit is meant to address.

### 2. Cheap Triage Before Expensive Assays

Real projects usually run cheap computational or low-cost screens before
expensive experiments.

MolForge equivalent:

- `evaluate_properties`
- `search_literature`
- `estimate_synthesizability`
- `dock_target`

These should be useful but imperfect. They help the model decide where to spend
more serious assay budget.

### 3. Expensive Evidence Gates

Real lead candidates require stronger evidence before nomination:

- potency evidence;
- toxicity/safety evidence;
- synthesis or route feasibility evidence;
- sometimes post-mutation or resistance-panel evidence.

MolForge equivalent:

- `assay_toxicity`
- `dock_target`
- `estimate_synthesizability`
- hard evidence requirements in `submit`
- `evidence_score`

This is why `submission_score` should remain strict. A molecule that looks good
but was never properly assayed is not a real lead candidate.

### 4. Cross-Functional Decision Board

Real projects are not controlled by one chemist. A lead-optimization meeting
usually includes:

- medicinal chemistry;
- assay biology;
- toxicology/safety;
- process chemistry or manufacturability;
- project leadership.

MolForge equivalent:

- `lead_chemist`
- `assay_planner`
- `toxicologist`
- `process_chemist`
- governance messages;
- hard vetoes;
- `coordination_score`

This is one of MolForge's strongest environment-innovation points. The agent is
not just optimizing a molecule; it is coordinating a scientific team.

### 5. Stop, Submit, or Restart

Real teams must decide when to stop spending money. Sometimes the right answer
is to abandon a scaffold early because the series is a trap.

MolForge equivalent:

- `submit`
- `restart`
- budget limits;
- max decision horizon;
- hard scenario target shift;
- sunk-cost trap in `level_2_hard`

This lets the environment test project judgment, not just local molecule edits.

## How To Use This In MolForge

### Keep Two Scores

Use two kinds of reward:

1. **Training reward**
   Helps the model learn the workflow.

2. **Formal submission score**
   Measures whether the agent actually nominated a valid candidate.

That means:

- `MOLFORGE_REWARD_MODE=curriculum` for early RL;
- default `assay_gated` mode for final reporting;
- `submission_score` stays `0.0` without a formal submit.

This mirrors the real world: a project can make progress without nominating a
lead, but it cannot claim lead success without a nomination package.

### Make Rewards Stage-Gated

A good real-world reward should not be one giant final number only.

Useful reward components:

- valid action/schema;
- useful design edit;
- useful first assay;
- evidence coverage;
- safety improvement;
- synthesis improvement;
- avoiding repeated assays;
- avoiding vetoed decisions;
- submitting only with enough support;
- restarting from a bad scaffold when appropriate.

This gives RL a learnable path while preserving strict final success.

### Make The Demo Story Simple

Judges should understand this in one sentence:

> MolForge tests whether an LLM can run a miniature drug-discovery project:
> design molecules, buy assays, respect safety vetoes, manage budget, and
> nominate a candidate only when the evidence package is strong enough.

Then show:

- baseline model repeats invalid or vetoed actions;
- SFT model learns the action language;
- RL model learns better evidence and submit timing;
- final candidate report card shows potency, toxicity, synthesis, evidence,
  budget, and coordination.

## What We Already Have

MolForge already contains most of this real-world structure:

- molecule slot edits;
- RDKit/TDC-backed surrogate oracle path;
- limited assay budget;
- cheap and expensive tools;
- hidden true properties;
- visible assay estimates;
- toxicity and synthesis constraints;
- multi-agent specialist governance;
- safety vetoes;
- restart action;
- hard target-shift scenario;
- decomposed report card;
- strict terminal `submission_score`;
- curriculum reward mode for early RL.

## What To Strengthen Next

The next useful additions should make the environment feel even more like a
real project:

1. **Assay uncertainty**
   Repeated assays should narrow confidence intervals, but cost budget.

2. **Stage labels**
   Mark states as `design`, `triage`, `evidence_package`, `nomination`, or
   `no-go`.

3. **No-go decisions**
   Reward a model for stopping or restarting when the evidence says the series
   is unsafe or infeasible.

4. **Portfolio-style report**
   At terminal time, show why the candidate was nominated or rejected.

5. **Holdout variants**
   Randomize scaffold starts and budgets so the model cannot memorize only
   three paths.

For the hackathon, the best near-term path is:

```text
SFT v4 for action/workflow competence
-> curriculum RL for observable reward improvement
-> strict assay_gated evaluation for final submission_score
-> README/demo framed as a real drug-discovery decision board
```

