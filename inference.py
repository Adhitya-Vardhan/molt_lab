"""Judge-facing baseline inference script for MolForge."""

from __future__ import annotations

import json
import os
from typing import Any, cast

from openai import OpenAI

from inference_common import (
    COMPACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_model_payload,
    extract_json,
)

try:
    from molforge.models import MolForgeAction, MolForgeObservation
    from molforge.server.molforge_environment import MolForgeEnvironment
except ImportError:
    from models import MolForgeAction, MolForgeObservation
    from server.molforge_environment import MolForgeEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TURNS = 10
MODEL_TIMEOUT_S = float(os.getenv("MODEL_TIMEOUT_S", "90"))
MODEL_LONG_TIMEOUT_S = float(os.getenv("MODEL_LONG_TIMEOUT_S", "150"))
MODEL_RETRY_TIMEOUT_S = float(os.getenv("MODEL_RETRY_TIMEOUT_S", "35"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "220"))


def main() -> None:
    env = MolForgeEnvironment()
    if not MODEL_NAME or not HF_TOKEN:
        raise RuntimeError("MODEL_NAME and HF_TOKEN are required. No heuristic fallback is available.")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores = []
    model_action_count = 0
    for episode_index in range(3):
        observation = env.reset()
        print(f"\n=== Episode {episode_index + 1}: {observation.scenario_id} ===")

        for _ in range(MAX_TURNS):
            if observation.done:
                break
            action = choose_action(client, observation)
            model_action_count += 1
            observation = env.step(action)
            print(
                f"step={observation.step_index:02d} action={action.action_type} actor={action.acting_role} "
                f"source=model reward={observation.reward:+.3f} budget={observation.remaining_budget} "
                f"governance={observation.governance.status} messages={len(action.messages)}"
            )
            print(f"  {observation.last_transition_summary}")
            if observation.done:
                break

        grader_scores = observation.metadata.get("terminal_grader_scores", {})
        submission_score = float(grader_scores.get("submission_score", 0.0))
        scores.append(submission_score)
        print(f"submission_score={submission_score:.3f}")
        if observation.report_card:
            print(observation.report_card)

    average = sum(scores) / len(scores)
    print("\n=== Baseline Summary ===")
    summary = {
        "scores": scores,
        "average_submission_score": round(average, 4),
        "model_action_count": model_action_count,
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "fallback_enabled": False,
    }
    print(json.dumps(summary, indent=2))


def choose_action(client: OpenAI, observation: MolForgeObservation) -> MolForgeAction:
    """Use the model and fail loudly when it cannot produce a valid action."""

    action, error = ask_model(client, observation)
    if action is None:
        raise RuntimeError(f"Model action failed: {error}")
    return action


def ask_model(client: OpenAI, observation: MolForgeObservation) -> tuple[Optional[MolForgeAction], str]:
    """Request a structured team action from the model and parse it safely."""

    errors = []
    try:
        full_payload = build_model_payload(observation, compact=False)
        timeout_s = model_timeout_for_step(observation)
        data = request_action_json(
            client=client,
            system_prompt=SYSTEM_PROMPT,
            user_payload=full_payload,
            timeout_s=timeout_s,
        )
        return MolForgeAction(**data), ""
    except Exception as exc:
        errors.append(f"full_prompt:{exc.__class__.__name__}:{exc}")
        try:
            compact_payload = build_model_payload(observation, compact=True)
            data = request_action_json(
                client=client,
                system_prompt=COMPACT_SYSTEM_PROMPT,
                user_payload=compact_payload,
                timeout_s=MODEL_RETRY_TIMEOUT_S,
            )
            return MolForgeAction(**data), ""
        except Exception as retry_exc:
            errors.append(f"compact_prompt:{retry_exc.__class__.__name__}:{retry_exc}")
            return None, " | ".join(errors)


def request_action_json(
    *,
    client: OpenAI,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Call the remote model with a bounded timeout and parse a JSON action."""

    configured_client = client.with_options(timeout=timeout_s)
    completion = configured_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=MODEL_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
    )
    message_content = completion.choices[0].message.content
    if isinstance(message_content, list):
        text = "".join(part.get("text", "") for part in cast(list[dict[str, Any]], message_content))
    else:
        text = message_content or ""
    return extract_json(text)


def model_timeout_for_step(observation: MolForgeObservation) -> float:
    """Allow more time for high-value late-stage decisions without making every step unbounded."""

    if observation.difficulty == "hard":
        return MODEL_LONG_TIMEOUT_S
    if observation.step_index >= observation.max_steps - 2:
        return MODEL_LONG_TIMEOUT_S
    return MODEL_TIMEOUT_S


if __name__ == "__main__":
    main()
