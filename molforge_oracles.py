"""RDKit/TDC-backed molecular oracle helpers for MolForge."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional


WARHEAD_SMILES = {
    "acrylamide": "C(=O)NC=C",
    "reversible_cyanoacrylamide": "C(=O)NC(=C)C#N",
    "nitrile": "C#N",
    "vinyl_sulfonamide": "S(=O)(=O)NC=C",
}

HINGE_SMILES = {
    "azaindole": "c1[nH]c2ccccc2n1",
    "pyridine": "c1ccncc1",
    "fluorophenyl": "c1ccc(F)cc1",
    "quinazoline": "c1ncnc2ccccc12",
}

TAIL_SMILES = {
    "morpholine": "N1CCOCC1",
    "piperazine": "N1CCNCC1",
    "cyclopropyl": "C1CC1",
    "dimethylamino": "N(C)C",
}

BACK_POCKET_SMILES = {
    "methoxy": "OC",
    "chloro": "Cl",
    "trifluoromethyl": "C(F)(F)F",
    "cyano": "C#N",
}

REFERENCE_LIGAND_SMILES = [
    "c%10(C(=O)NC=C)c(c1ccncc1)c(C1CC1)c(OC)cc%10",
    "c%10(C#N)c(c1ccncc1)c(N1CCOCC1)c(C#N)cc%10",
    "c%10(C(=O)NC=C)c(c1ccc(F)cc1)c(N(C)C)c(Cl)cc%10",
]


def assemble_surrogate_smiles(molecule: Mapping[str, str]) -> str:
    """Build a valid substituted-aryl SMILES for RDKit/TDC scoring."""

    return (
        f"c%10({WARHEAD_SMILES[molecule['warhead']]})"
        f"c({HINGE_SMILES[molecule['hinge']]})"
        f"c({TAIL_SMILES[molecule['solvent_tail']]})"
        f"c({BACK_POCKET_SMILES[molecule['back_pocket']]})cc%10"
    )


def oracle_backend_status() -> Dict[str, bool]:
    """Report which external chemistry engines are importable."""

    return {"rdkit": _rdkit_modules() is not None, "tdc": _tdc_oracle_class() is not None}


def evaluate_with_rdkit_tdc(
    molecule: Mapping[str, str],
    fallback_properties: Mapping[str, float],
) -> Dict[str, float]:
    """Blend RDKit/TDC medicinal-chemistry signals into MolForge properties."""

    modules = _rdkit_modules()
    if modules is None:
        return dict(fallback_properties)

    Chem = modules["Chem"]
    Descriptors = modules["Descriptors"]
    Crippen = modules["Crippen"]
    Lipinski = modules["Lipinski"]
    QED = modules["QED"]
    rdFingerprintGenerator = modules["rdFingerprintGenerator"]
    rdMolDescriptors = modules["rdMolDescriptors"]
    DataStructs = modules["DataStructs"]

    smiles = assemble_surrogate_smiles(molecule)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(fallback_properties)
    canonical = Chem.MolToSmiles(mol)

    qed_value = _tdc_oracle_score("QED", canonical)
    if qed_value is None:
        qed_value = float(QED.qed(mol))
    qed_score = _clamp01(qed_value)

    sa_value = _tdc_oracle_score("SA", canonical)
    synth_score = _normalize_sa(sa_value)
    if synth_score is None:
        synth_score = _rdkit_synth_proxy(mol, Descriptors, Lipinski, rdMolDescriptors)

    logp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    mol_wt = float(Descriptors.MolWt(mol))
    rotatable = float(Lipinski.NumRotatableBonds(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    reference_similarity = _reference_ligand_similarity(mol, Chem, rdFingerprintGenerator, DataStructs)
    diagnostics = chemistry_diagnostics(molecule)
    chemical_quality = float(diagnostics["chemical_quality"])

    property_risk = _property_risk(logp=logp, tpsa=tpsa, mol_wt=mol_wt, rotatable=rotatable)
    structural_risk = _structural_alert_risk(molecule)
    rdkit_toxicity = _clamp01(0.55 * property_risk + 0.45 * structural_risk)

    target_fit = _target_fit_proxy(
        molecule,
        qed_score=qed_score,
        logp=logp,
        tpsa=tpsa,
        aromatic_rings=aromatic_rings,
    )
    anchored_target_fit = _clamp01(0.78 * target_fit + 0.22 * reference_similarity)
    novelty = _novelty_proxy(mol, Chem, rdFingerprintGenerator, DataStructs)
    chemistry_pressure = max(0.0, 0.58 - chemical_quality)

    return {
        "potency": round(
            _clamp01(_blend(fallback_properties["potency"], anchored_target_fit, 0.38) - 0.06 * chemistry_pressure),
            4,
        ),
        "safety": round(
            _clamp01(1.0 - _blend(fallback_properties["toxicity"], rdkit_toxicity + 0.10 * chemistry_pressure, 0.25)),
            4,
        ),
        "toxicity": round(_clamp01(_blend(fallback_properties["toxicity"], rdkit_toxicity, 0.25) + 0.10 * chemistry_pressure), 4),
        "synth": round(_blend(fallback_properties["synth"], min(synth_score, chemical_quality), 0.55), 4),
        "novelty": round(_blend(fallback_properties["novelty"], novelty, 0.50), 4),
        "chemical_quality": round(chemical_quality, 4),
        "reference_similarity": round(reference_similarity, 4),
    }


def chemistry_diagnostics(molecule: Mapping[str, str]) -> Dict[str, Any]:
    """Return descriptor windows, alert flags, and anchor similarity for a molecule."""

    modules = _rdkit_modules()
    if modules is None:
        return {
            "available": False,
            "chemical_quality": 0.5,
            "passes_filters": True,
            "reference_similarity": 0.5,
            "alerts": [],
            "failed_filters": [],
        }

    Chem = modules["Chem"]
    Descriptors = modules["Descriptors"]
    Crippen = modules["Crippen"]
    Lipinski = modules["Lipinski"]
    rdFingerprintGenerator = modules["rdFingerprintGenerator"]
    rdMolDescriptors = modules["rdMolDescriptors"]
    DataStructs = modules["DataStructs"]

    smiles = assemble_surrogate_smiles(molecule)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "available": False,
            "chemical_quality": 0.0,
            "passes_filters": False,
            "reference_similarity": 0.0,
            "alerts": ["invalid_smiles"],
            "failed_filters": ["rdkit_parse"],
        }

    mol_wt = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    rotatable = float(Lipinski.NumRotatableBonds(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    reference_similarity = _reference_ligand_similarity(mol, Chem, rdFingerprintGenerator, DataStructs)

    property_window_score = _mean(
        [
            _soft_window_score(mol_wt, lower=260.0, upper=460.0, tolerance=85.0),
            _soft_window_score(logp, lower=1.5, upper=4.5, tolerance=1.1),
            _soft_window_score(tpsa, lower=35.0, upper=110.0, tolerance=30.0),
            _soft_upper_bound_score(hbd, upper=3.0, tolerance=1.5),
            _soft_upper_bound_score(hba, upper=8.0, tolerance=2.0),
            _soft_upper_bound_score(rotatable, upper=8.0, tolerance=3.0),
            _soft_window_score(aromatic_rings, lower=1.0, upper=4.0, tolerance=1.0),
            _soft_window_score(fraction_csp3, lower=0.05, upper=0.45, tolerance=0.15),
        ]
    )
    alert_names = _medicinal_chemistry_alerts(molecule)
    pains_alerts = _pains_alerts(mol)
    alert_names.extend(pains_alerts)
    alert_penalty = min(1.0, 0.18 * len(alert_names))
    failed_filters = []

    if not 240.0 <= mol_wt <= 500.0:
        failed_filters.append("molecular_weight")
    if not 1.0 <= logp <= 5.1:
        failed_filters.append("logp")
    if not 25.0 <= tpsa <= 125.0:
        failed_filters.append("tpsa")
    if hbd > 4.0:
        failed_filters.append("hbd")
    if hba > 10.0:
        failed_filters.append("hba")
    if rotatable > 9.0:
        failed_filters.append("rotatable_bonds")
    if len(alert_names) > 2:
        failed_filters.append("alert_count")

    chemical_quality = _clamp01(
        0.48 * property_window_score
        + 0.22 * (1.0 - alert_penalty)
        + 0.20 * _soft_upper_bound_score(len(alert_names), upper=1.0, tolerance=2.0)
        + 0.10 * reference_similarity
    )
    passes_filters = chemical_quality >= 0.58 and not failed_filters

    return {
        "available": True,
        "smiles": smiles,
        "chemical_quality": round(chemical_quality, 4),
        "reference_similarity": round(reference_similarity, 4),
        "passes_filters": passes_filters,
        "alerts": sorted(set(alert_names)),
        "failed_filters": failed_filters,
        "descriptors": {
            "mol_wt": round(mol_wt, 3),
            "logp": round(logp, 3),
            "tpsa": round(tpsa, 3),
            "hbd": round(hbd, 3),
            "hba": round(hba, 3),
            "rotatable_bonds": round(rotatable, 3),
            "aromatic_rings": round(aromatic_rings, 3),
            "fraction_csp3": round(fraction_csp3, 3),
        },
    }


@lru_cache(maxsize=1)
def _rdkit_modules() -> Optional[Dict[str, Any]]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors
    except Exception:
        return None
    return {
        "Chem": Chem,
        "Crippen": Crippen,
        "DataStructs": DataStructs,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
        "QED": QED,
        "rdFingerprintGenerator": rdFingerprintGenerator,
        "rdMolDescriptors": rdMolDescriptors,
    }


@lru_cache(maxsize=1)
def _tdc_oracle_class() -> Optional[Any]:
    try:
        from tdc import Oracle
    except Exception:
        return None
    return Oracle


@lru_cache(maxsize=8)
def _tdc_oracle(name: str) -> Optional[Any]:
    oracle_class = _tdc_oracle_class()
    if oracle_class is None:
        return None
    try:
        return oracle_class(name=name)
    except Exception:
        return None


def _tdc_oracle_score(name: str, smiles: str) -> Optional[float]:
    oracle = _tdc_oracle(name)
    if oracle is None:
        return None
    try:
        value = oracle(smiles)
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_sa(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return _clamp01(value)
    return _clamp01((10.0 - value) / 9.0)


def _rdkit_synth_proxy(mol: Any, Descriptors: Any, Lipinski: Any, rdMolDescriptors: Any) -> float:
    mol_wt = float(Descriptors.MolWt(mol))
    rotatable = float(Lipinski.NumRotatableBonds(mol))
    stereocenters = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    ring_count = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    complexity = (
        max(0.0, mol_wt - 350.0) / 260.0
        + rotatable / 12.0
        + stereocenters / 4.0
        + max(0.0, ring_count - 3.0) / 4.0
        + aromatic_rings / 8.0
    )
    return _clamp01(1.0 - 0.35 * complexity)


def _property_risk(*, logp: float, tpsa: float, mol_wt: float, rotatable: float) -> float:
    logp_risk = _sigmoid((logp - 3.5) / 1.15)
    size_risk = _sigmoid((mol_wt - 500.0) / 90.0)
    flexibility_risk = _sigmoid((rotatable - 8.0) / 2.5)
    polarity_risk = _sigmoid((tpsa - 130.0) / 32.0)
    return _clamp01(0.42 * logp_risk + 0.24 * size_risk + 0.20 * flexibility_risk + 0.14 * polarity_risk)


def _structural_alert_risk(molecule: Mapping[str, str]) -> float:
    risk = 0.18
    if molecule["warhead"] == "acrylamide":
        risk += 0.12
    if molecule["warhead"] == "vinyl_sulfonamide":
        risk += 0.22
    if molecule["solvent_tail"] == "dimethylamino":
        risk += 0.24
    if molecule["back_pocket"] == "trifluoromethyl":
        risk += 0.20
    if molecule["hinge"] == "fluorophenyl" and molecule["back_pocket"] in {"chloro", "trifluoromethyl"}:
        risk += 0.12
    if molecule["solvent_tail"] in {"morpholine", "piperazine"}:
        risk -= 0.08
    if molecule["warhead"] == "nitrile":
        risk -= 0.08
    return _clamp01(risk)


def _target_fit_proxy(
    molecule: Mapping[str, str],
    *,
    qed_score: float,
    logp: float,
    tpsa: float,
    aromatic_rings: float,
) -> float:
    lipophilic_match = 1.0 - min(abs(logp - 3.0) / 4.0, 1.0)
    polarity_match = 1.0 - min(abs(tpsa - 85.0) / 110.0, 1.0)
    pocket_match = 0.0
    if molecule["hinge"] in {"azaindole", "quinazoline"}:
        pocket_match += 0.18
    if molecule["back_pocket"] in {"cyano", "chloro", "trifluoromethyl"}:
        pocket_match += 0.14
    if molecule["warhead"] in {"acrylamide", "reversible_cyanoacrylamide", "nitrile"}:
        pocket_match += 0.12
    if aromatic_rings >= 2:
        pocket_match += 0.08
    return _clamp01(0.20 + 0.30 * lipophilic_match + 0.22 * polarity_match + 0.18 * qed_score + pocket_match)


def _novelty_proxy(mol: Any, Chem: Any, rdFingerprintGenerator: Any, DataStructs: Any) -> float:
    similarities = _reference_similarities(mol, Chem, rdFingerprintGenerator, DataStructs)
    if not similarities:
        return 0.5
    return _clamp01(1.0 - max(similarities))


def _reference_ligand_similarity(mol: Any, Chem: Any, rdFingerprintGenerator: Any, DataStructs: Any) -> float:
    similarities = _reference_similarities(mol, Chem, rdFingerprintGenerator, DataStructs)
    if not similarities:
        return 0.5
    return _clamp01(max(similarities))


def _reference_similarities(mol: Any, Chem: Any, rdFingerprintGenerator: Any, DataStructs: Any) -> list[float]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = generator.GetFingerprint(mol)
    similarities = []
    for ref in REFERENCE_LIGAND_SMILES:
        ref_mol = Chem.MolFromSmiles(ref)
        if ref_mol is None:
            continue
        ref_fp = generator.GetFingerprint(ref_mol)
        similarities.append(float(DataStructs.TanimotoSimilarity(fp, ref_fp)))
    return similarities


def _medicinal_chemistry_alerts(molecule: Mapping[str, str]) -> list[str]:
    alerts: list[str] = []
    if molecule["warhead"] == "acrylamide":
        alerts.append("michael_acceptor")
    if molecule["warhead"] == "vinyl_sulfonamide":
        alerts.extend(["strong_michael_acceptor", "sulfonamide_reactivity"])
    if molecule["solvent_tail"] == "dimethylamino":
        alerts.append("basic_tertiary_amine")
    if molecule["back_pocket"] == "trifluoromethyl":
        alerts.append("lipophilic_alert")
    if molecule["hinge"] == "fluorophenyl" and molecule["back_pocket"] in {"chloro", "trifluoromethyl"}:
        alerts.append("polyhalogenated_hydrophobe")
    return alerts


def _pains_alerts(mol: Any) -> list[str]:
    catalog = _filter_catalog()
    if catalog is None:
        return []
    try:
        matches = catalog.GetMatches(mol)
    except Exception:
        return []
    alert_names = []
    for match in matches:
        try:
            alert_names.append(f"catalog:{match.GetDescription()}")
        except Exception:
            continue
    return alert_names[:3]


@lru_cache(maxsize=1)
def _filter_catalog() -> Optional[Any]:
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    except Exception:
        return None
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        return FilterCatalog(params)
    except Exception:
        return None


def _blend(fallback_value: float, oracle_value: float, oracle_weight: float) -> float:
    return _clamp01((1.0 - oracle_weight) * fallback_value + oracle_weight * oracle_value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _soft_window_score(value: float, *, lower: float, upper: float, tolerance: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return _clamp01(1.0 - (lower - value) / max(tolerance, 1e-6))
    return _clamp01(1.0 - (value - upper) / max(tolerance, 1e-6))


def _soft_upper_bound_score(value: float, *, upper: float, tolerance: float) -> float:
    if value <= upper:
        return 1.0
    return _clamp01(1.0 - (value - upper) / max(tolerance, 1e-6))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
