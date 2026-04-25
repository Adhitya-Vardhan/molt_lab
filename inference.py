"""Judge-facing baseline inference script for MolForge."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, cast

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

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MAX_TURNS = 10
MODEL_TIMEOUT_S = float(os.getenv("MODEL_TIMEOUT_S", "35"))
MODEL_LONG_TIMEOUT_S = float(os.getenv("MODEL_LONG_TIMEOUT_S", "45"))
MODEL_RETRY_TIMEOUT_S = float(os.getenv("MODEL_RETRY_TIMEOUT_S", "15"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "220"))
MIN_REPORTED_SCORE = 1e-6
MAX_REPORTED_SCORE = 1.0 - 1e-6


def main() -> None:
    env = MolForgeEnvironment()
    if not API_BASE_URL or not MODEL_NAME or not API_KEY:
        raise RuntimeError(
            "API_BASE_URL, MODEL_NAME, and API_KEY or HF_TOKEN are required. "
            "No heuristic fallback is available."
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = []
    raw_final_scores = []
    submission_scores = []
    progress_scores = []
    model_action_count = 0
    for episode_index in range(3):
        observation = env.reset()
        task_name = observation.scenario_id
        episode_error = ""
        print(
            f"[START] task={task_name} difficulty={observation.difficulty} episode={episode_index + 1}",
            flush=True,
        )

        for _ in range(MAX_TURNS):
            if observation.done:
                break
            try:
                action = choose_action(client, observation)
                model_action_count += 1
                observation = env.step(action)
            except Exception as exc:
                episode_error = f"{exc.__class__.__name__}:{exc}"
                print(
                    f"[STEP] task={task_name} step={observation.step_index + 1} "
                    f"reward=0.000000 action=model_error status=failed",
                    flush=True,
                )
                break
            print(
                f"[STEP] task={task_name} step={observation.step_index} "
                f"reward={observation.reward:.6f} action={action.action_type} "
                f"actor={action.acting_role} status={observation.governance.status}",
                flush=True,
            )
            if observation.done:
                break

        grader_scores = observation.metadata.get("terminal_grader_scores", {})
        raw_final_score = float(grader_scores.get("final_score", grader_scores.get("submission_score", 0.0)))
        final_score = reportable_score(raw_final_score)
        submission_score = float(grader_scores.get("submission_score", 0.0))
        progress_score = float(grader_scores.get("progress_score", 0.0))
        scores.append(final_score)
        raw_final_scores.append(raw_final_score)
        submission_scores.append(submission_score)
        progress_scores.append(progress_score)
        end_line = (
            f"[END] task={task_name} score={final_score:.6f} raw_score={raw_final_score:.6f} "
            f"submission_score={submission_score:.6f} progress_score={progress_score:.6f} "
            f"steps={observation.step_index}"
        )
        if episode_error:
            end_line += f" error={json.dumps(episode_error)}"
        print(end_line, flush=True)
        if observation.report_card:
            print(observation.report_card, flush=True)

    average = sum(scores) / len(scores)
    average_progress = sum(progress_scores) / len(progress_scores)
    summary = {
        "scores": scores,
        "raw_final_scores": raw_final_scores,
        "average_final_score": round(reportable_score(average), 6),
        "submission_scores": submission_scores,
        "average_submission_score": round(sum(submission_scores) / len(submission_scores), 4),
        "progress_scores": progress_scores,
        "average_progress_score": round(average_progress, 4),
        "model_action_count": model_action_count,
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "fallback_enabled": False,
    }
    print("[SUMMARY] " + json.dumps(summary, separators=(",", ":")), flush=True)


def reportable_score(score: float) -> float:
    """Validator-facing scores must be strictly between 0 and 1."""

    if score <= 0.0:
        return MIN_REPORTED_SCORE
    if score >= 1.0:
        return MAX_REPORTED_SCORE
    return score


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
