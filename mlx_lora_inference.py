"""MLX-backed local LoRA inference runner for MolForge on Apple Silicon."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from inference_common import (
    COMPACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    attach_team_messages,
    build_model_payload,
    extract_json,
)

try:
    from molforge.models import MolForgeAction, MolForgeObservation
    from molforge.server.molforge_environment import MolForgeEnvironment
except ImportError:
    from models import MolForgeAction, MolForgeObservation
    from server.molforge_environment import MolForgeEnvironment


ADAPTER_PATH = Path(os.getenv("LORA_ADAPTER_PATH", "qwen3_5_2b_lora_adapters_strict"))
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/Qwen3.5-2B")
LOCAL_NUM_EPISODES = int(os.getenv("LOCAL_NUM_EPISODES", "3"))
LOCAL_MAX_TURNS = int(os.getenv("LOCAL_MAX_TURNS", "10"))
MLX_MAX_TOKENS = int(os.getenv("MLX_MAX_TOKENS", "768"))
MLX_RETRY_MAX_TOKENS = int(os.getenv("MLX_RETRY_MAX_TOKENS", "512"))
MLX_JSON_PREFILL = os.getenv("MLX_JSON_PREFILL", "true").lower() == "true"
MLX_COMPACT_ACTION = os.getenv("MLX_COMPACT_ACTION", "false").lower() == "true"
MLX_COMPACT_REPAIR = os.getenv("MLX_COMPACT_REPAIR", "false").lower() == "true"
MLX_FORCED_ACTION_TYPES = [
    item.strip()
    for item in os.getenv("MLX_FORCED_ACTION_TYPES", "").split(",")
    if item.strip()
]
JSON_PREFILL = '{"action_type":"'
COMPACT_ACTION_SYSTEM_PROMPT = """
You control the MolForge action policy.
Return exactly one JSON object with only these top-level keys:
action_type, acting_role, edit_type, slot, fragment, tool_name, rationale,
evidence, expected_effects.

Valid action_type values are exactly:
edit, run_assay, submit, restart, defer.

Do not output team messages. Do not output proposal, approval, objection,
risk_flag, assay_request, rejection, or submission_recommendation as action_type.
The environment will attach governance messages automatically.

Role rules:
- run_assay uses acting_role "assay_planner" and a valid tool_name.
- edit, submit, restart, and defer use acting_role "lead_chemist".
- unused optional fields must be JSON null.
""".strip()


def main() -> None:
    adapter_path = ADAPTER_PATH.expanduser().resolve()
    print(f"Using MLX base model: {BASE_MODEL_NAME}", flush=True)
    print(f"Using LoRA adapter: {adapter_path}", flush=True)
    model, tokenizer = load(BASE_MODEL_NAME, adapter_path=str(adapter_path))
    sampler = make_sampler(temp=0.0)

    env = MolForgeEnvironment()
    scores = []
    submission_scores = []
    progress_scores = []

    for episode_index in range(LOCAL_NUM_EPISODES):
        observation = env.reset()
        print(f"\n=== Episode {episode_index + 1}: {observation.scenario_id} ===", flush=True)

        for _ in range(LOCAL_MAX_TURNS):
            if observation.done:
                break
            action, source, elapsed = choose_mlx_action(model, tokenizer, sampler, observation)
            if MLX_COMPACT_ACTION:
                action = attach_team_messages(observation, action)
            observation = env.step(action)
            print(
                f"step={observation.step_index:02d} action={action.action_type} actor={action.acting_role} "
                f"source={source} gen_s={elapsed:.2f} reward={observation.reward:+.3f} "
                f"budget={observation.remaining_budget} governance={observation.governance.status}",
                flush=True,
            )
            print(f"  {observation.last_transition_summary}", flush=True)
            if observation.done:
                break

        grader_scores = observation.metadata.get("terminal_grader_scores", {})
        final_score = float(grader_scores.get("final_score", grader_scores.get("submission_score", 0.0)))
        submission_score = float(grader_scores.get("submission_score", 0.0))
        progress_score = float(grader_scores.get("progress_score", 0.0))
        scores.append(final_score)
        submission_scores.append(submission_score)
        progress_scores.append(progress_score)
        print(f"final_score={final_score:.3f}", flush=True)
        print(f"submission_score={submission_score:.3f}", flush=True)
        print(f"progress_score={progress_score:.3f}", flush=True)
        if observation.report_card:
            print(observation.report_card, flush=True)

    average = sum(scores) / len(scores)
    average_progress = sum(progress_scores) / len(progress_scores)
    print("\n=== MLX LoRA Local Summary ===", flush=True)
    print(
        json.dumps(
            {
                "adapter": str(adapter_path),
                "base_model": BASE_MODEL_NAME,
                "scores": scores,
                "average_final_score": round(average, 4),
                "submission_scores": submission_scores,
                "average_submission_score": round(sum(submission_scores) / len(submission_scores), 4),
                "progress_scores": progress_scores,
                "average_progress_score": round(average_progress, 4),
            },
            indent=2,
        ),
        flush=True,
    )


def choose_mlx_action(
    model,
    tokenizer,
    sampler,
    observation: MolForgeObservation,
) -> Tuple[MolForgeAction, str, float]:
    started = time.perf_counter()
    action, error = ask_mlx_model(
        model,
        tokenizer,
        sampler,
        observation,
        compact=False,
        max_tokens=MLX_MAX_TOKENS,
        forced_action_type=None,
    )
    if action is not None:
        return action, "mlx_lora_model", time.perf_counter() - started

    forced_errors = []
    for forced_action_type in forced_action_types(observation):
        forced_action, forced_error = ask_mlx_model(
            model,
            tokenizer,
            sampler,
            observation,
            compact=True,
            max_tokens=MLX_RETRY_MAX_TOKENS,
            forced_action_type=forced_action_type,
        )
        if forced_action is not None:
            return (
                forced_action,
                f"mlx_lora_forced_{forced_action_type}",
                time.perf_counter() - started,
            )
        forced_errors.append(f"{forced_action_type}:{forced_error}")

    retry_action, retry_error = ask_mlx_model(
        model,
        tokenizer,
        sampler,
        observation,
        compact=True,
        max_tokens=MLX_RETRY_MAX_TOKENS,
        forced_action_type=None,
    )
    if retry_action is not None:
        return retry_action, "mlx_lora_compact_retry", time.perf_counter() - started

    raise RuntimeError(
        "MLX LoRA action failed: "
        f"full_prompt:{error} | forced:{' || '.join(forced_errors)} | compact_prompt:{retry_error}"
    )


def ask_mlx_model(
    model,
    tokenizer,
    sampler,
    observation: MolForgeObservation,
    *,
    compact: bool,
    max_tokens: int,
    forced_action_type: Optional[str],
) -> Tuple[Optional[MolForgeAction], str]:
    response_text = ""
    try:
        payload = (
            compact_action_payload(observation)
            if MLX_COMPACT_ACTION
            else build_model_payload(observation, compact=compact)
        )
        system_prompt = (
            COMPACT_ACTION_SYSTEM_PROMPT
            if MLX_COMPACT_ACTION
            else (COMPACT_SYSTEM_PROMPT if compact else SYSTEM_PROMPT)
        )
        response_text = generate_response(
            model,
            tokenizer,
            sampler,
            system_prompt=system_prompt,
            user_payload=payload,
            max_tokens=max_tokens,
            use_json_prefill=MLX_JSON_PREFILL,
            forced_action_type=forced_action_type,
        )
        if MLX_JSON_PREFILL:
            response_text = json_prefill(forced_action_type) + response_text
        data = extract_json(response_text)
        repair_notes: list[str] = []
        if MLX_COMPACT_ACTION and MLX_COMPACT_REPAIR:
            data, repair_notes = repair_compact_action(data)
        if MLX_COMPACT_ACTION and "messages" in data:
            raise ValueError("compact action output must not include messages")
        action = MolForgeAction(**data)
        if repair_notes:
            action.metadata["compact_repair_notes"] = repair_notes
        return action, ""
    except Exception as exc:
        snippet = response_text[:1200].replace("\n", "\\n")
        return None, f"{exc.__class__.__name__}:{exc}; raw={snippet}"


def generate_response(
    model,
    tokenizer,
    sampler,
    *,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_tokens: int,
    use_json_prefill: bool,
    forced_action_type: Optional[str],
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if use_json_prefill:
        prompt += json_prefill(forced_action_type)
    return generate(
        model,
        tokenizer,
        prompt,
        verbose=False,
        max_tokens=max_tokens,
        sampler=sampler,
    ).strip()


def json_prefill(forced_action_type: Optional[str]) -> str:
    if forced_action_type:
        return f'{{"action_type":"{forced_action_type}",'
    return JSON_PREFILL


def forced_action_types(observation: MolForgeObservation) -> list[str]:
    if MLX_FORCED_ACTION_TYPES:
        return MLX_FORCED_ACTION_TYPES
    if observation.step_index == 0:
        if observation.scenario_id == "level_2_hard":
            return ["restart", "edit", "run_assay", "defer"]
        return ["edit", "run_assay", "defer"]
    return ["run_assay", "edit", "submit", "restart", "defer"]


def compact_action_payload(observation: MolForgeObservation) -> dict[str, Any]:
    lead_view = next(
        (role.observation for role in observation.role_observations if role.role == "lead_chemist"),
        {},
    )
    assay_view = next(
        (role.observation for role in observation.role_observations if role.role == "assay_planner"),
        {},
    )
    return {
        "valid_action_types": ["edit", "run_assay", "submit", "restart", "defer"],
        "scenario_id": observation.scenario_id,
        "difficulty": observation.difficulty,
        "task_brief": observation.task_brief,
        "current_molecule": observation.current_molecule,
        "current_smiles": observation.metadata.get("current_smiles", ""),
        "visible_metrics": observation.visible_metrics,
        "constraint_status": [constraint.model_dump() for constraint in observation.constraint_status],
        "remaining_budget": observation.remaining_budget,
        "max_budget": observation.max_budget,
        "step_index": observation.step_index,
        "max_steps": observation.max_steps,
        "molecule_slots": lead_view.get("molecule_slots", {}),
        "candidate_edits": lead_view.get("candidate_edits", [])[:12],
        "open_questions": lead_view.get("open_questions", []),
        "known_assays": [
            {
                "tool_name": reading.tool_name,
                "property_name": reading.property_name,
                "estimate": reading.estimate,
                "confidence_low": reading.confidence_low,
                "confidence_high": reading.confidence_high,
                "molecule_signature": reading.molecule_signature,
            }
            for reading in observation.known_assays[-8:]
        ],
        "tool_costs": assay_view.get("tool_costs", {}),
        "evidence_gaps": assay_view.get("evidence_gaps", []),
        "estimated_information_value": assay_view.get("estimated_information_value", {}),
    }


def repair_compact_action(data: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Bounded normalization for compact-action models.

    This repairs only schema-near-misses. It does not invent an action from a
    non-action wrapper and it still rejects invalid top-level action types.
    """

    repaired = dict(data)
    notes: list[str] = []

    if "role" in repaired and "acting_role" not in repaired:
        repaired["acting_role"] = repaired.pop("role")
        notes.append("role->acting_role")

    action_type = repaired.get("action_type")
    if action_type not in {"edit", "run_assay", "submit", "restart", "defer"}:
        return repaired, notes

    if repaired.get("edit_type") == "replace":
        repaired["edit_type"] = "substitute"
        notes.append("edit_type:replace->substitute")

    if isinstance(repaired.get("evidence"), str):
        repaired["evidence"] = [repaired["evidence"]]
        notes.append("evidence:string->list")

    repaired["expected_effects"] = repair_effects(repaired.get("expected_effects"), notes)

    if action_type == "run_assay":
        repaired["acting_role"] = "assay_planner"
        repaired["edit_type"] = None
        repaired["slot"] = None
        repaired["fragment"] = None
        if repaired.get("tool_name") not in {
            "evaluate_properties",
            "dock_target",
            "assay_toxicity",
            "estimate_synthesizability",
            "evaluate_novelty",
            "search_literature",
            "run_md_simulation",
        }:
            repaired["tool_name"] = "evaluate_properties"
            notes.append("tool_name:invalid->evaluate_properties")
    else:
        repaired["acting_role"] = "lead_chemist"
        if action_type == "edit":
            if repaired.get("edit_type") not in {"add_fragment", "substitute", "remove", "undo_last_edit"}:
                repaired["edit_type"] = "substitute"
                notes.append("edit_type:invalid->substitute")
            if repaired.get("tool_name") is not None:
                repaired["tool_name"] = None
                notes.append("tool_name:edit->null")
        else:
            for key in ("edit_type", "slot", "fragment", "tool_name"):
                if repaired.get(key) is not None:
                    repaired[key] = None
                    notes.append(f"{key}:{action_type}->null")

    allowed_keys = {
        "action_type",
        "acting_role",
        "edit_type",
        "slot",
        "fragment",
        "tool_name",
        "rationale",
        "evidence",
        "expected_effects",
    }
    for key in list(repaired):
        if key not in allowed_keys:
            repaired.pop(key)
            notes.append(f"drop_extra:{key}")

    repaired.setdefault("rationale", "Choose the next compact MolForge action.")
    repaired.setdefault("evidence", [])
    for key in ("edit_type", "slot", "fragment", "tool_name"):
        repaired.setdefault(key, None)

    return repaired, notes


def repair_effects(value: Any, notes: list[str]) -> dict[str, str]:
    defaults = {
        "potency": "unknown",
        "toxicity": "unknown",
        "synth": "unknown",
        "novelty": "unknown",
        "budget": "neutral",
    }
    if not isinstance(value, dict):
        notes.append("expected_effects:non_dict->defaults")
        return defaults

    aliases = {
        "synthesizability": "synth",
        "synthesis": "synth",
    }
    for raw_key, raw_value in value.items():
        key = aliases.get(raw_key, raw_key)
        if key not in defaults:
            notes.append(f"expected_effects:drop_extra:{raw_key}")
            continue
        defaults[key] = normalize_effect_value(raw_value, notes, key)
    return defaults


def normalize_effect_value(value: Any, notes: list[str], key: str) -> str:
    if value in {"up", "down", "neutral", "unknown", "not_applicable"}:
        return value
    text = str(value).lower().strip().replace("-", "_").replace(" ", "_")
    if any(token in text for token in ("increase", "improve", "higher", "upward", "+")):
        notes.append(f"expected_effects:{key}:{value}->up")
        return "up"
    if any(token in text for token in ("decrease", "lower", "reduce", "downward", "-")):
        notes.append(f"expected_effects:{key}:{value}->down")
        return "down"
    if any(token in text for token in ("maintain", "stable", "unchanged", "same")):
        notes.append(f"expected_effects:{key}:{value}->neutral")
        return "neutral"
    if "not_applicable" in text or text == "na":
        notes.append(f"expected_effects:{key}:{value}->not_applicable")
        return "not_applicable"
    notes.append(f"expected_effects:{key}:{value}->unknown")
    return "unknown"


if __name__ == "__main__":
    main()
