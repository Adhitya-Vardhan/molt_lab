"""Local-only inference runner for Ollama-backed MolForge testing.

This script is intentionally separate from `inference.py`.
Use `inference.py` for the judge-facing OpenAI-client baseline required by the
hackathon. Use this file for local development against Ollama's native API,
where reasoning models often behave better when `think` is explicitly disabled.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import requests

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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "gemma4:e2b")
LOCAL_NUM_EPISODES = int(os.getenv("LOCAL_NUM_EPISODES", "3"))
LOCAL_MAX_TURNS = int(os.getenv("LOCAL_MAX_TURNS", "10"))
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "240"))
OLLAMA_RETRY_TIMEOUT_S = float(os.getenv("OLLAMA_RETRY_TIMEOUT_S", "120"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "768"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"


def main() -> None:
    env = MolForgeEnvironment()
    scores = []

    print(f"Using Ollama model: {LOCAL_MODEL_NAME}", flush=True)
    print(f"Ollama base URL: {OLLAMA_BASE_URL}", flush=True)
    print(f"Thinking enabled: {OLLAMA_THINK}", flush=True)

    for episode_index in range(LOCAL_NUM_EPISODES):
        observation = env.reset()
        print(f"\n=== Episode {episode_index + 1}: {observation.scenario_id} ===", flush=True)

        for _ in range(LOCAL_MAX_TURNS):
            if observation.done:
                break
            action, source = choose_local_action(observation)
            observation = env.step(action)
            print(
                f"step={observation.step_index:02d} action={action.action_type} actor={action.acting_role} "
                f"source={source} reward={observation.reward:+.3f} budget={observation.remaining_budget} "
                f"governance={observation.governance.status}",
                flush=True,
            )
            print(f"  {observation.last_transition_summary}", flush=True)
            if observation.done:
                break

        grader_scores = observation.metadata.get("terminal_grader_scores", {})
        submission_score = float(grader_scores.get("submission_score", 0.0))
        scores.append(submission_score)
        print(f"submission_score={submission_score:.3f}", flush=True)
        if observation.report_card:
            print(observation.report_card, flush=True)

    average = sum(scores) / len(scores)
    print("\n=== Local Baseline Summary ===", flush=True)
    print(
        json.dumps(
            {
                "model": LOCAL_MODEL_NAME,
                "scores": scores,
                "average_submission_score": round(average, 4),
            },
            indent=2,
        ),
        flush=True,
    )


def choose_local_action(observation: MolForgeObservation) -> Tuple[MolForgeAction, str]:
    """Use Ollama output and fail loudly if it cannot produce a valid action."""

    action, error = ask_ollama_model(observation)
    if action is not None:
        return action, "model"
    raise RuntimeError(f"Local model action failed: {error}")


def ask_ollama_model(observation: MolForgeObservation) -> Tuple[Optional[MolForgeAction], str]:
    """Call Ollama's native chat API.

    Official Ollama docs note that reasoning traces live in `message.thinking`
    while the final answer lives in `message.content`, and that `think: false`
    can disable thinking on the native chat endpoint.
    """

    errors = []
    try:
        payload = build_model_payload(observation, compact=False)
        response_json = ollama_chat(
            system_prompt=SYSTEM_PROMPT,
            user_payload=payload,
            timeout_s=OLLAMA_TIMEOUT_S,
        )
        data = parse_ollama_json_response(response_json)
        return MolForgeAction(**data), ""
    except Exception as exc:
        errors.append(f"full_prompt:{exc.__class__.__name__}:{exc}")
        try:
            payload = build_model_payload(observation, compact=True)
            response_json = ollama_chat(
                system_prompt=COMPACT_SYSTEM_PROMPT,
                user_payload=payload,
                timeout_s=OLLAMA_RETRY_TIMEOUT_S,
            )
            data = parse_ollama_json_response(response_json)
            return MolForgeAction(**data), ""
        except Exception as retry_exc:
            errors.append(f"compact_prompt:{retry_exc.__class__.__name__}:{retry_exc}")
            return None, " | ".join(errors)


def ollama_chat(
    *,
    system_prompt: str,
    user_payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    """Issue a native Ollama chat request."""

    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": LOCAL_MODEL_NAME,
            "stream": False,
            "think": OLLAMA_THINK,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_MAX_TOKENS,
            },
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def parse_ollama_json_response(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a JSON action from a native Ollama response."""

    message = response_json.get("message", {}) or {}
    content = message.get("content", "") or ""
    thinking = message.get("thinking", "") or ""

    if content:
        try:
            return extract_json(content)
        except Exception:
            pass

    if thinking:
        try:
            return extract_json(thinking)
        except Exception:
            pass

    combined = f"{content}\n{thinking}".strip()
    if combined:
        return extract_json(combined)

    raise ValueError("No parseable JSON action found in Ollama response")


if __name__ == "__main__":
    main()
