import random

from molforge.scenarios import SCENARIOS, build_scenario_variant, molecule_diagnostics
from molforge.server.molforge_environment import MolForgeEnvironment


def test_holdout_variant_generation_is_deterministic():
    base = SCENARIOS[0]

    left = build_scenario_variant(
        base,
        rng=random.Random("holdout-seed"),
        variant_kind="holdout",
        variant_index=7,
    )
    right = build_scenario_variant(
        base,
        rng=random.Random("holdout-seed"),
        variant_kind="holdout",
        variant_index=7,
    )

    assert left == right
    assert left.variant_kind == "holdout"
    assert left.scenario_family == base.scenario_id
    assert left.scenario_id != base.scenario_id
    assert left.starting_scaffold != base.starting_scaffold
    assert abs(sum(left.objective_weights.values()) - 1.0) < 1e-6


def test_chemistry_diagnostics_prefer_realistic_molecules():
    good = molecule_diagnostics(
        {
            "warhead": "nitrile",
            "hinge": "azaindole",
            "solvent_tail": "morpholine",
            "back_pocket": "cyano",
        }
    )
    bad = molecule_diagnostics(
        {
            "warhead": "vinyl_sulfonamide",
            "hinge": "quinazoline",
            "solvent_tail": "dimethylamino",
            "back_pocket": "trifluoromethyl",
        }
    )

    assert good["available"] is True
    assert good["passes_filters"] is True
    assert good["chemical_quality"] > bad["chemical_quality"]
    assert bad["passes_filters"] is False
    assert bad["alerts"]


def test_holdout_environment_surfaces_variant_metadata(monkeypatch):
    monkeypatch.setenv("MOLFORGE_SCENARIO_MODE", "holdout")
    monkeypatch.setenv("MOLFORGE_RANDOM_SEED", "test-seed")

    env = MolForgeEnvironment()
    obs = env.reset()

    variant = obs.metadata["scenario_variant"]
    assert variant["mode"] == "holdout"
    assert variant["is_holdout"] is True
    assert variant["is_randomized"] is True
    assert obs.metadata["chemical_diagnostics"]["available"] is True
    assert "chemical_quality" in obs.visible_metrics
