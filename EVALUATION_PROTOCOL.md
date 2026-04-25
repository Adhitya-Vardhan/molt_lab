# MolForge Evaluation Protocol

Use two reward settings for different purposes.

## 1. Training / RL Warmup

Use curriculum mode:

```bash
MOLFORGE_REWARD_MODE=curriculum
MOLFORGE_TRAINING_RANDOMIZATION=1
```

Track:

- mean episode reward;
- valid JSON/action rate;
- policy veto rate;
- evidence score;
- number of oracle calls;
- budget remaining at submit;
- submit rate;
- missed-nomination rate;
- strict terminal `submission_score`.

Curriculum reward is allowed to be generous because its purpose is learning.
It rewards useful evidence collection and evidence-supported submit timing.

## 2. Judge-Facing Evaluation

Use strict/default mode:

```bash
unset MOLFORGE_TRAINING_RANDOMIZATION
export MOLFORGE_REWARD_MODE=assay_gated
```

Report:

- `average_submission_score`;
- per-task `submission_score`;
- `candidate_score`;
- `constraint_margin_score`;
- `evidence_score`;
- `coordination_score`;
- `budget_score`;
- submitted vs not submitted;
- invalid action count;
- policy veto count.

The official score should not be minimum number of steps. Real drug discovery
does not reward the fastest project if it skips necessary evidence. Instead,
MolForge rewards finishing within the available budget and decision horizon.

## Budget And Step Interpretation

MolForge has both:

- `max_steps`: the project decision deadline;
- `remaining_budget`: the assay/resource budget.

The agent must finish inside both limits.

Budget effects:

- assays subtract from `remaining_budget`;
- over-budget assays are invalid;
- budget exhaustion terminates the episode;
- valid submissions receive a transition-level `budget_efficiency` reward;
- formal `submission_score` receives a small bonus for unused budget only when
  the submission has required evidence, passes constraints, and beats baseline;
- curriculum near-miss reward includes `budget_score`, but missed nomination is
  penalized if the evidence package was ready and the model failed to submit.

Step effects:

- reaching `max_steps` without submission ends the episode;
- there is a step-limit penalty;
- no extra score is given merely for fewer steps;
- faster is better only if the candidate is supported by evidence and budget is
  preserved.

## Recommended Comparison Table

For the README/demo, compare:

| Model | Reward mode | Submit rate | Avg submission_score | Avg evidence_score | Avg budget_score | Veto rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base model | assay_gated | low | low | low/medium | variable | high |
| SFT v4 | assay_gated | better | better | better | variable | lower |
| SFT v4 + RL | assay_gated | best | best | high | healthy | low |

For training plots, show curriculum reward increasing, but always pair it with
strict `submission_score` before/after so the improvement is credible.

