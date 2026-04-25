# MolForge Compact Policy V4 Dataset

This dataset is the next SFT warm-start set for Qwen3.5-2B-class models.

Path:

```text
issue/molforge_sft_compact_policy_v4.jsonl
```

## Why V4 Exists

The v3 adapter produced valid compact actions sometimes, but in evaluation it
got `submission_score=0.0` on easy, medium, and hard. The failures were mostly
policy failures, not only JSON failures:

- it sometimes needed repair or forced decoding to produce valid JSON;
- it repeated toxicologist-vetoed edits;
- it did not recover after a blocked action;
- it often failed to submit even when a useful candidate had been built;
- it did not reliably use restart in the hard trap scenario.

V4 trains against those exact issues.

## What Changed

V4 uses the same compact system prompt and user payload shape as
`mlx_lora_inference.py`. This matters because v3 trained on a nested dataset
payload that did not exactly match inference.

The assistant target is action-only:

```json
{
  "action_type": "edit",
  "acting_role": "lead_chemist",
  "edit_type": "substitute",
  "slot": "solvent_tail",
  "fragment": "morpholine",
  "tool_name": null,
  "rationale": "...",
  "evidence": ["..."],
  "expected_effects": {
    "potency": "unknown",
    "toxicity": "down",
    "synth": "up",
    "novelty": "unknown",
    "budget": "neutral"
  }
}
```

No `messages` field is included. The runner attaches governance messages
deterministically before calling `env.step(...)`.

## Dataset Summary

Generated record count: `4284`

Action coverage:

- `edit`: `1382`
- `run_assay`: `2254`
- `submit`: `455`
- `restart`: `157`
- `defer`: `36`

Tool coverage:

- `assay_toxicity`: `642`
- `estimate_synthesizability`: `639`
- `dock_target`: `613`
- `evaluate_properties`: `108`
- `evaluate_novelty`: `108`
- `search_literature`: `108`
- `run_md_simulation`: `36`

Validation checks passed:

- no top-level `proposal`;
- no nested `messages`;
- no `edit_type="replace"`;
- `evidence` is always a list;
- `expected_effects` always has exactly `potency`, `toxicity`, `synth`,
  `novelty`, and `budget`;
- every assistant target parses as `MolForgeAction`.

## How This Affects RL

SFT should not be treated as the final optimizer. Its job is to make the model
competent enough that RL receives meaningful trajectories:

1. Valid JSON/action schema.
2. Reasonable first moves.
3. Recovery after veto.
4. Evidence collection before submission.
5. Hard-scenario restart behavior.
6. Formal submit when evidence gates are satisfied.

The previous v3 adapter got `submission_score=0.0` on all three tested
episodes, although it did receive some non-terminal step rewards. That is a bad
starting point for sparse terminal RL because the model rarely reaches the
reward-rich submit state.

With v4 SFT, the goal is not to get a perfect model immediately. The goal is to
make the model reach valid submit/restart/evidence patterns often enough that
GRPO/RL can show an upward reward curve.

For the hackathon judging criteria, this gives a clean story:

- baseline model: valid-action and policy failures;
- SFT v4: schema and workflow competence improves;
- RL: measurable reward improvement from environment feedback.

