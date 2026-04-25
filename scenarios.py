"""Scenario configs and RDKit/TDC-backed surrogate chemistry for MolForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping


SLOT_ORDER = ["warhead", "hinge", "solvent_tail", "back_pocket"]
EDITABLE_SLOTS = ["warhead", "hinge", "solvent_tail", "back_pocket"]


@dataclass(frozen=True)
class FragmentSpec:
    """Per-fragment surrogate property contributions."""

    name: str
    potency: float
    safety: float
    synth: float
    novelty: float
    literature_hint: str


@dataclass(frozen=True)
class ScenarioConfig:
    """Single evaluation scenario."""

    scenario_id: str
    difficulty: str
    target_name: str
    task_brief: str
    oracle_budget: int
    max_steps: int
    starting_scaffold: Mapping[str, str]
    restart_scaffold: Mapping[str, str]
    objective_weights: Mapping[str, float]
    hard_constraints: Mapping[str, float]
    target_shift_step: int | None = None
    trap_penalty: bool = False
    enabled_tools: List[str] = field(default_factory=list)
    enabled_actions: List[str] = field(default_factory=list)
    coordination_mode: str = "multi_agent"
    enabled_roles: List[str] = field(default_factory=list)
    required_review_roles: List[str] = field(default_factory=list)
    max_messages_per_turn: int = 4
    baseline_to_beat: float = 0.5


FRAGMENT_LIBRARY: Dict[str, Dict[str, FragmentSpec]] = {
    "warhead": {
        "acrylamide": FragmentSpec(
            "acrylamide",
            potency=0.18,
            safety=-0.03,
            synth=0.02,
            novelty=0.03,
            literature_hint="Covalent warheads often boost KRAS potency but can increase reactivity risk.",
        ),
        "reversible_cyanoacrylamide": FragmentSpec(
            "reversible_cyanoacrylamide",
            potency=0.16,
            safety=0.06,
            synth=-0.04,
            novelty=0.08,
            literature_hint="Reversible covalent warheads can preserve potency while softening safety liabilities.",
        ),
        "nitrile": FragmentSpec(
            "nitrile",
            potency=0.11,
            safety=0.09,
            synth=0.05,
            novelty=0.04,
            literature_hint="Nitrile warheads are safer but may need stronger pocket complementarity to keep potency.",
        ),
        "vinyl_sulfonamide": FragmentSpec(
            "vinyl_sulfonamide",
            potency=0.13,
            safety=-0.07,
            synth=-0.05,
            novelty=0.10,
            literature_hint="Sulfonamide warheads can be potent but often pressure synthesis and safety.",
        ),
    },
    "hinge": {
        "azaindole": FragmentSpec(
            "azaindole",
            potency=0.17,
            safety=0.01,
            synth=-0.03,
            novelty=0.06,
            literature_hint="Azaindoles are strong binders in KRAS-like pockets when the warhead is well aligned.",
        ),
        "pyridine": FragmentSpec(
            "pyridine",
            potency=0.10,
            safety=0.04,
            synth=0.05,
            novelty=0.02,
            literature_hint="Simple heteroaryl hinges improve tractability and keep synthesis accessible.",
        ),
        "fluorophenyl": FragmentSpec(
            "fluorophenyl",
            potency=0.12,
            safety=-0.08,
            synth=0.04,
            novelty=0.03,
            literature_hint="Hydrophobic hinge binders can lift affinity while increasing lipophilic liability.",
        ),
        "quinazoline": FragmentSpec(
            "quinazoline",
            potency=0.15,
            safety=-0.04,
            synth=-0.06,
            novelty=0.05,
            literature_hint="Quinazolines are potent but can create a heavy, synthesis-taxing scaffold.",
        ),
    },
    "solvent_tail": {
        "morpholine": FragmentSpec(
            "morpholine",
            potency=0.06,
            safety=0.16,
            synth=0.07,
            novelty=0.02,
            literature_hint="Morpholine tails frequently de-risk hERG and improve solubility.",
        ),
        "piperazine": FragmentSpec(
            "piperazine",
            potency=0.05,
            safety=0.10,
            synth=0.03,
            novelty=0.03,
            literature_hint="Basic cyclic tails improve polarity but can trigger clearance concerns if overused.",
        ),
        "cyclopropyl": FragmentSpec(
            "cyclopropyl",
            potency=0.08,
            safety=-0.03,
            synth=0.04,
            novelty=0.04,
            literature_hint="Compact hydrophobes sometimes improve fit but rarely help safety.",
        ),
        "dimethylamino": FragmentSpec(
            "dimethylamino",
            potency=0.04,
            safety=-0.13,
            synth=0.02,
            novelty=0.04,
            literature_hint="Strongly basic tails can quickly create cardiac and CNS liabilities.",
        ),
    },
    "back_pocket": {
        "methoxy": FragmentSpec(
            "methoxy",
            potency=0.07,
            safety=0.08,
            synth=0.06,
            novelty=0.02,
            literature_hint="Small polar back-pocket groups often stabilize potency without blowing up toxicity.",
        ),
        "chloro": FragmentSpec(
            "chloro",
            potency=0.12,
            safety=-0.12,
            synth=0.04,
            novelty=0.02,
            literature_hint="Halogens often buy potency at the cost of lipophilic risk.",
        ),
        "trifluoromethyl": FragmentSpec(
            "trifluoromethyl",
            potency=0.14,
            safety=-0.15,
            synth=-0.02,
            novelty=0.06,
            literature_hint="CF3 groups can strongly improve affinity but frequently over-shoot safety windows.",
        ),
        "cyano": FragmentSpec(
            "cyano",
            potency=0.10,
            safety=0.03,
            synth=0.01,
            novelty=0.05,
            literature_hint="Cyano groups are efficient potency handles when hydrophobic groups are too risky.",
        ),
    },
}

DEFAULT_TOOL_COSTS: Dict[str, int] = {
    "evaluate_properties": 50,
    "search_literature": 100,
    "dock_target": 300,
    "estimate_synthesizability": 120,
    "evaluate_novelty": 80,
    "assay_toxicity": 2000,
    "run_md_simulation": 2500,
}


SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        scenario_id="level_0_easy",
        difficulty="easy",
        target_name="KRAS G12C",
        task_brief=(
            "Improve target potency while repairing a mild safety liability and keeping synthesis "
            "evidence current. The starting scaffold is close, but a strong submission still needs "
            "the right edit sequence plus assay support."
        ),
        oracle_budget=3600,
        max_steps=7,
        starting_scaffold={
            "warhead": "acrylamide",
            "hinge": "pyridine",
            "solvent_tail": "cyclopropyl",
            "back_pocket": "chloro",
        },
        restart_scaffold={
            "warhead": "reversible_cyanoacrylamide",
            "hinge": "pyridine",
            "solvent_tail": "morpholine",
            "back_pocket": "methoxy",
        },
        objective_weights={
            "potency": 0.55,
            "safety": 0.15,
            "synth": 0.15,
            "novelty": 0.15,
        },
        hard_constraints={"potency_min": 0.84, "toxicity_max": 0.28, "synth_min": 0.62},
        enabled_tools=list(DEFAULT_TOOL_COSTS.keys()),
        enabled_actions=["edit", "run_assay", "submit", "defer", "restart"],
        enabled_roles=[
            "lead_chemist",
            "toxicologist",
            "assay_planner",
            "process_chemist",
        ],
        required_review_roles=["toxicologist", "assay_planner", "process_chemist"],
        baseline_to_beat=0.70,
    ),
    ScenarioConfig(
        scenario_id="level_1_medium",
        difficulty="medium",
        target_name="KRAS G12C",
        task_brief=(
            "Balance potency, toxicity, and synthesizability under budget pressure. The best "
            "molecules require coordinated safety edits plus current assay evidence."
        ),
        oracle_budget=4300,
        max_steps=8,
        starting_scaffold={
            "warhead": "acrylamide",
            "hinge": "fluorophenyl",
            "solvent_tail": "dimethylamino",
            "back_pocket": "chloro",
        },
        restart_scaffold={
            "warhead": "reversible_cyanoacrylamide",
            "hinge": "azaindole",
            "solvent_tail": "morpholine",
            "back_pocket": "cyano",
        },
        objective_weights={
            "potency": 0.42,
            "safety": 0.33,
            "synth": 0.13,
            "novelty": 0.12,
        },
        hard_constraints={"potency_min": 0.76, "toxicity_max": 0.34, "synth_min": 0.62},
        enabled_tools=list(DEFAULT_TOOL_COSTS.keys()),
        enabled_actions=["edit", "run_assay", "submit", "defer", "restart"],
        enabled_roles=[
            "lead_chemist",
            "toxicologist",
            "assay_planner",
            "process_chemist",
        ],
        required_review_roles=["toxicologist", "assay_planner", "process_chemist"],
        baseline_to_beat=0.64,
    ),
    ScenarioConfig(
        scenario_id="level_2_hard",
        difficulty="hard",
        target_name="KRAS G12C resistance panel",
        task_brief=(
            "Solve a non-stationary design problem with a fixed, problematic core. The starting "
            "series is a sunk-cost trap, and the target pocket shifts late in the episode."
        ),
        oracle_budget=5000,
        max_steps=9,
        starting_scaffold={
            "warhead": "acrylamide",
            "hinge": "quinazoline",
            "solvent_tail": "dimethylamino",
            "back_pocket": "trifluoromethyl",
        },
        restart_scaffold={
            "warhead": "nitrile",
            "hinge": "azaindole",
            "solvent_tail": "morpholine",
            "back_pocket": "cyano",
        },
        objective_weights={
            "potency": 0.38,
            "safety": 0.32,
            "synth": 0.16,
            "novelty": 0.14,
        },
        hard_constraints={"potency_min": 0.78, "toxicity_max": 0.46, "synth_min": 0.62},
        target_shift_step=4,
        trap_penalty=True,
        enabled_tools=list(DEFAULT_TOOL_COSTS.keys()),
        enabled_actions=["edit", "run_assay", "submit", "defer", "restart"],
        enabled_roles=[
            "lead_chemist",
            "toxicologist",
            "assay_planner",
            "process_chemist",
        ],
        required_review_roles=["toxicologist", "assay_planner", "process_chemist"],
        baseline_to_beat=0.66,
    ),
]

SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}


def get_scenario(index: int) -> ScenarioConfig:
    """Return scenarios in a stable cycle so repeated resets cover all tasks."""

    return SCENARIOS[index % len(SCENARIOS)]


def format_molecule(molecule: Mapping[str, str]) -> str:
    """Human-readable canonical representation."""

    ordered = [f"{slot}={molecule[slot]}" for slot in SLOT_ORDER]
    return " | ".join(ordered)


def fragment_choices(slot: str) -> List[str]:
    """Return the editable fragments for a slot."""

    return sorted(FRAGMENT_LIBRARY[slot].keys())


def evaluate_molecule(
    molecule: Mapping[str, str],
    scenario: ScenarioConfig,
    *,
    target_shift_active: bool = False,
) -> Dict[str, float]:
    """Evaluate a molecule with target logic plus RDKit/TDC medicinal chemistry signals."""

    potency = 0.23
    safety = 0.56
    synth = 0.58
    novelty = 0.18

    for slot, fragment_name in molecule.items():
        fragment = FRAGMENT_LIBRARY[slot][fragment_name]
        potency += fragment.potency
        safety += fragment.safety
        synth += fragment.synth
        novelty += fragment.novelty

    if molecule["warhead"] == "acrylamide" and molecule["hinge"] == "azaindole":
        potency += 0.10
    if molecule["solvent_tail"] == "morpholine" and molecule["back_pocket"] == "methoxy":
        safety += 0.08
    if molecule["hinge"] == "fluorophenyl" and molecule["back_pocket"] == "chloro":
        potency += 0.06
        safety -= 0.16
    if molecule["solvent_tail"] == "dimethylamino" and molecule["back_pocket"] == "trifluoromethyl":
        safety -= 0.15
    if molecule["warhead"] == "nitrile" and molecule["back_pocket"] == "cyano":
        potency += 0.04
        novelty += 0.03
    if molecule["warhead"] == "reversible_cyanoacrylamide" and molecule["solvent_tail"] == "morpholine":
        safety += 0.05

    if target_shift_active:
        if molecule["warhead"] == "acrylamide":
            potency -= 0.16
        if molecule["warhead"] == "nitrile":
            potency += 0.10
        if molecule["back_pocket"] == "cyano":
            potency += 0.03

    if scenario.trap_penalty:
        potency = min(potency, 0.71)
        safety = min(safety, 0.44)

    potency = min(max(potency, 0.0), 1.0)
    safety = min(max(safety, 0.0), 1.0)
    synth = min(max(synth, 0.0), 1.0)
    novelty = min(max(novelty, 0.0), 1.0)
    toxicity = min(max(1.0 - safety, 0.0), 1.0)

    fallback_properties = {
        "potency": round(potency, 4),
        "safety": round(safety, 4),
        "toxicity": round(toxicity, 4),
        "synth": round(synth, 4),
        "novelty": round(novelty, 4),
    }
    try:
        from molforge_oracles import evaluate_with_rdkit_tdc
    except Exception:
        return fallback_properties
    return evaluate_with_rdkit_tdc(molecule, fallback_properties)


def molecule_to_smiles(molecule: Mapping[str, str]) -> str:
    """Return the RDKit/TDC surrogate SMILES used by the chemistry oracle."""

    try:
        from molforge_oracles import assemble_surrogate_smiles
    except Exception:
        return ""
    return assemble_surrogate_smiles(molecule)


def oracle_backend_status() -> Dict[str, bool]:
    """Return whether RDKit and TDC are active for scoring."""

    try:
        from molforge_oracles import oracle_backend_status as backend_status
    except Exception:
        return {"rdkit": False, "tdc": False}
    return backend_status()


def compute_objective_score(properties: Mapping[str, float], scenario: ScenarioConfig) -> float:
    """Aggregate visible scientific goals into a single 0-1 quality score."""

    safety_score = 1.0 - properties["toxicity"]
    score = (
        scenario.objective_weights["potency"] * properties["potency"]
        + scenario.objective_weights["safety"] * safety_score
        + scenario.objective_weights["synth"] * properties["synth"]
        + scenario.objective_weights["novelty"] * properties["novelty"]
    )
    return round(min(max(score, 0.0), 1.0), 4)


def evaluate_constraints(
    properties: Mapping[str, float], scenario: ScenarioConfig
) -> Dict[str, tuple[bool, float]]:
    """Return hard-constraint satisfaction results."""

    results: Dict[str, tuple[bool, float]] = {}
    if "potency_min" in scenario.hard_constraints:
        threshold = scenario.hard_constraints["potency_min"]
        results["potency_min"] = (properties["potency"] >= threshold, threshold)
    if "toxicity_max" in scenario.hard_constraints:
        threshold = scenario.hard_constraints["toxicity_max"]
        results["toxicity_max"] = (properties["toxicity"] <= threshold, threshold)
    if "synth_min" in scenario.hard_constraints:
        threshold = scenario.hard_constraints["synth_min"]
        results["synth_min"] = (properties["synth"] >= threshold, threshold)
    return results


def evaluate_constraint_margins(
    properties: Mapping[str, float], scenario: ScenarioConfig
) -> Dict[str, float]:
    """Return proportional 0-1 constraint scores where larger violations score lower."""

    margins: Dict[str, float] = {}
    if "potency_min" in scenario.hard_constraints:
        threshold = scenario.hard_constraints["potency_min"]
        margins["potency_min"] = min(1.0, max(0.0, properties["potency"] / max(threshold, 1e-6)))
    if "toxicity_max" in scenario.hard_constraints:
        threshold = scenario.hard_constraints["toxicity_max"]
        if properties["toxicity"] <= threshold:
            margins["toxicity_max"] = 1.0
        else:
            excess = properties["toxicity"] - threshold
            margins["toxicity_max"] = max(0.0, 1.0 - excess / max(1.0 - threshold, 1e-6))
    if "synth_min" in scenario.hard_constraints:
        threshold = scenario.hard_constraints["synth_min"]
        margins["synth_min"] = min(1.0, max(0.0, properties["synth"] / max(threshold, 1e-6)))
    return margins


def literature_hints(molecule: Mapping[str, str]) -> List[str]:
    """Collect deterministic medicinal chemistry hints for the current molecule."""

    hints = []
    for slot in SLOT_ORDER:
        fragment_name = molecule[slot]
        hints.append(FRAGMENT_LIBRARY[slot][fragment_name].literature_hint)
    return hints


def enumerate_candidate_edits(molecule: Mapping[str, str]) -> Iterable[tuple[str, str]]:
    """Generate all single-edit candidates from the current molecule."""

    for slot in SLOT_ORDER:
        for fragment in fragment_choices(slot):
            if molecule[slot] != fragment:
                yield slot, fragment
