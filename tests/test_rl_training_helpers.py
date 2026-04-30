import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_submission.rl_training_helpers import build_dynamic_prompts, evaluate_completion


def test_dynamic_prompts_store_full_pre_actions():
    dataset = build_dynamic_prompts(episodes=1, max_turns=2, randomized=False, seed="unit-test")
    rows = list(dataset)

    assert rows
    later_rows = [row for row in rows if row["record"]["pre_actions"]]
    assert later_rows

    first_action = later_rows[0]["record"]["pre_actions"][0]
    assert "action_type" in first_action
    assert "acting_role" in first_action
    assert "messages" in first_action
    assert "evidence" in first_action
    assert "expected_effects" in first_action


def test_reward_shaping_no_longer_prefers_assay_over_good_edit():
    record = {"pre_actions": [], "randomized": False, "random_seed": "unit-test"}

    edit_completion = json.dumps(
        {
            "action_type": "edit",
            "acting_role": "lead_chemist",
            "edit_type": "substitute",
            "slot": "solvent_tail",
            "fragment": "morpholine",
            "tool_name": None,
            "rationale": "Switch to morpholine to improve safety and tractability.",
            "evidence": [],
            "expected_effects": {
                "potency": "neutral",
                "toxicity": "down",
                "synth": "up",
                "novelty": "neutral",
                "budget": "neutral",
            },
        }
    )
    assay_completion = json.dumps(
        {
            "action_type": "run_assay",
            "acting_role": "assay_planner",
            "edit_type": None,
            "slot": None,
            "fragment": None,
            "tool_name": "evaluate_properties",
            "rationale": "Run a quick assay before changing the molecule.",
            "evidence": [],
            "expected_effects": {
                "potency": "unknown",
                "toxicity": "unknown",
                "synth": "unknown",
                "novelty": "unknown",
                "budget": "down",
            },
        }
    )

    edit_reward, _ = evaluate_completion("", edit_completion, record)
    assay_reward, _ = evaluate_completion("", assay_completion, record)

    assert edit_reward > assay_reward
