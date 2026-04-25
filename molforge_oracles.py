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
    novelty = _novelty_proxy(mol, Chem, rdFingerprintGenerator, DataStructs)

    return {
        "potency": round(_blend(fallback_properties["potency"], target_fit, 0.35), 4),
        "safety": round(_clamp01(1.0 - _blend(fallback_properties["toxicity"], rdkit_toxicity, 0.25)), 4),
        "toxicity": round(_blend(fallback_properties["toxicity"], rdkit_toxicity, 0.25), 4),
        "synth": round(_blend(fallback_properties["synth"], synth_score, 0.55), 4),
        "novelty": round(_blend(fallback_properties["novelty"], novelty, 0.50), 4),
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
    refs = [
        "c%10(C(=O)NC=C)c(c1ccncc1)c(C1CC1)c(OC)cc%10",
        "c%10(C#N)c(c1ccncc1)c(N1CCOCC1)c(C#N)cc%10",
        "c%10(C(=O)NC=C)c(c1ccc(F)cc1)c(N(C)C)c(Cl)cc%10",
    ]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = generator.GetFingerprint(mol)
    similarities = []
    for ref in refs:
        ref_mol = Chem.MolFromSmiles(ref)
        if ref_mol is None:
            continue
        ref_fp = generator.GetFingerprint(ref_mol)
        similarities.append(float(DataStructs.TanimotoSimilarity(fp, ref_fp)))
    if not similarities:
        return 0.5
    return _clamp01(1.0 - max(similarities))


def _blend(fallback_value: float, oracle_value: float, oracle_weight: float) -> float:
    return _clamp01((1.0 - oracle_weight) * fallback_value + oracle_weight * oracle_value)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
