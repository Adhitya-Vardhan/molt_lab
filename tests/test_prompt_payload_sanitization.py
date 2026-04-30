import sys
from pathlib import Path

from molforge.server.molforge_environment import MolForgeEnvironment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_common import build_model_payload
from mlx_lora_inference import compact_action_payload
from scripts.generate_sft_compact_policy_v4_dataset import compact_action_payload as dataset_compact_action_payload


FORBIDDEN_KEY_FRAGMENTS = {
    "reward",
    "score",
    "grader",
    "report_card",
    "terminal_grader_scores",
}


def _flatten_keys(value, prefix=""):
    keys = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]"
            keys.extend(_flatten_keys(nested, path))
    return keys


def test_full_model_payload_excludes_scoring_fields():
    env = MolForgeEnvironment()
    observation = env.reset()

    payload = build_model_payload(observation, compact=False)
    flat_keys = _flatten_keys(payload)

    assert "visible_metrics" in payload
    assert "constraint_status" in payload
    assert "known_assays" in payload
    assert not any(
        any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
        for key in flat_keys
    )


def test_compact_model_payload_excludes_scoring_fields():
    env = MolForgeEnvironment()
    observation = env.reset()

    payload = build_model_payload(observation, compact=True)
    flat_keys = _flatten_keys(payload)

    assert "visible_metrics" in payload
    assert "constraint_status" in payload
    assert not any(
        any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
        for key in flat_keys
    )


def test_mlx_compact_action_payload_excludes_policy_heuristic_scores():
    env = MolForgeEnvironment()
    observation = env.reset()

    payload = compact_action_payload(observation)
    flat_keys = _flatten_keys(payload)

    assert "evidence_gaps" in payload
    assert "tool_costs" in payload
    assert "estimated_information_value" not in payload
    assert not any(
        any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
        for key in flat_keys
    )


def test_dataset_compact_action_payload_excludes_policy_heuristic_scores():
    env = MolForgeEnvironment()
    observation = env.reset()

    payload = dataset_compact_action_payload(observation)
    flat_keys = _flatten_keys(payload)

    assert "evidence_gaps" in payload
    assert "tool_costs" in payload
    assert "estimated_information_value" not in payload
    assert not any(
        any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
        for key in flat_keys
    )
