"""Local PEFT/LoRA inference runner for MolForge.

Use this to test an SFT adapter against the environment before RL. It loads the
base model named in the adapter config, attaches the LoRA weights, and requires
the model to emit a valid MolForgeAction JSON object. There is no heuristic
fallback or schema repair.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

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


ADAPTER_PATH = Path(os.getenv("LORA_ADAPTER_PATH", "qwen3_5_2b_lora_adapters"))
LOCAL_NUM_EPISODES = int(os.getenv("LOCAL_NUM_EPISODES", "3"))
LOCAL_MAX_TURNS = int(os.getenv("LOCAL_MAX_TURNS", "10"))
LORA_MAX_NEW_TOKENS = int(os.getenv("LORA_MAX_NEW_TOKENS", "768"))
LORA_RETRY_MAX_NEW_TOKENS = int(os.getenv("LORA_RETRY_MAX_NEW_TOKENS", "512"))
LORA_DEVICE = os.getenv("LORA_DEVICE", "auto")


def main() -> None:
    adapter_path = ADAPTER_PATH.expanduser().resolve()
    tokenizer, model, base_model_name, device = load_adapter_model(adapter_path)
    env = MolForgeEnvironment()
    scores = []

    print(f"Using LoRA adapter: {adapter_path}", flush=True)
    print(f"Base model: {base_model_name}", flush=True)
    print(f"Device: {device}", flush=True)

    for episode_index in range(LOCAL_NUM_EPISODES):
        observation = env.reset()
        print(f"\n=== Episode {episode_index + 1}: {observation.scenario_id} ===", flush=True)

        for _ in range(LOCAL_MAX_TURNS):
            if observation.done:
                break
            action, source = choose_lora_action(tokenizer, model, observation, device)
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
    print("\n=== LoRA Local Summary ===", flush=True)
    print(
        json.dumps(
            {
                "adapter": str(adapter_path),
                "base_model": base_model_name,
                "scores": scores,
                "average_submission_score": round(average, 4),
            },
            indent=2,
        ),
        flush=True,
    )


def load_adapter_model(adapter_path: Path):
    config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = config.base_model_name_or_path
    device = resolve_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    model_class = (
        Qwen3_5ForConditionalGeneration
        if "Qwen3_5ForConditionalGeneration" in (base_config.architectures or [])
        else AutoModelForCausalLM
    )
    base_model = model_class.from_pretrained(
        base_model_name,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model, base_model_name, device


def resolve_device() -> str:
    if LORA_DEVICE != "auto":
        return LORA_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_lora_action(
    tokenizer,
    model,
    observation: MolForgeObservation,
    device: str,
) -> Tuple[MolForgeAction, str]:
    action, error = ask_lora_model(
        tokenizer,
        model,
        observation,
        device,
        compact=False,
        max_new_tokens=LORA_MAX_NEW_TOKENS,
    )
    if action is not None:
        return action, "lora_model"

    retry_action, retry_error = ask_lora_model(
        tokenizer,
        model,
        observation,
        device,
        compact=True,
        max_new_tokens=LORA_RETRY_MAX_NEW_TOKENS,
    )
    if retry_action is not None:
        return retry_action, "lora_model_compact_retry"

    raise RuntimeError(f"LoRA model action failed: full_prompt:{error} | compact_prompt:{retry_error}")


def ask_lora_model(
    tokenizer,
    model,
    observation: MolForgeObservation,
    device: str,
    *,
    compact: bool,
    max_new_tokens: int,
) -> Tuple[Optional[MolForgeAction], str]:
    response_text = ""
    try:
        payload = build_model_payload(observation, compact=compact)
        system_prompt = COMPACT_SYSTEM_PROMPT if compact else SYSTEM_PROMPT
        response_text = generate_response(
            tokenizer,
            model,
            device,
            system_prompt=system_prompt,
            user_payload=payload,
            max_new_tokens=max_new_tokens,
        )
        data = extract_json(response_text)
        return MolForgeAction(**data), ""
    except Exception as exc:
        snippet = response_text[:1200].replace("\n", "\\n")
        return None, f"{exc.__class__.__name__}:{exc}; raw={snippet}"


def generate_response(
    tokenizer,
    model,
    device: str,
    *,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_new_tokens: int,
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    main()
