"""Shared imports, constants, and utility mixins for MolForge."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional

try:
    from ..models import (
        AgentMessage,
        AssayReading,
        MolForgeAction,
    )
    from ..scenarios import (
        DEFAULT_TOOL_COSTS,
        EDITABLE_SLOTS,
        FRAGMENT_LIBRARY,
        SLOT_ORDER,
        SCENARIOS,
        ScenarioConfig,
        build_scenario_variant,
        compute_objective_score,
        molecule_diagnostics,
        enumerate_candidate_edits,
        evaluate_constraint_margins,
        evaluate_constraints,
        evaluate_molecule,
        format_molecule,
        get_scenario,
        literature_hints,
        molecule_to_smiles,
        oracle_backend_status,
    )
except ImportError:
    from models import (
        AgentMessage,
        AssayReading,
        MolForgeAction,
    )
    from scenarios import (
        DEFAULT_TOOL_COSTS,
        EDITABLE_SLOTS,
        FRAGMENT_LIBRARY,
        SLOT_ORDER,
        SCENARIOS,
        ScenarioConfig,
        build_scenario_variant,
        compute_objective_score,
        molecule_diagnostics,
        enumerate_candidate_edits,
        evaluate_constraint_margins,
        evaluate_constraints,
        evaluate_molecule,
        format_molecule,
        get_scenario,
        literature_hints,
        molecule_to_smiles,
        oracle_backend_status,
    )


ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "lead_chemist": ["edit", "submit", "restart", "defer"],
    "toxicologist": [],
    "assay_planner": ["run_assay"],
    "process_chemist": [],
}

ROLE_MESSAGE_TYPES: Dict[str, List[str]] = {
    "lead_chemist": ["proposal", "revision_request", "submission_recommendation"],
    "toxicologist": ["approval", "objection", "risk_flag", "assay_request", "rejection"],
    "assay_planner": ["proposal", "approval", "rejection", "assay_request", "submission_recommendation"],
    "process_chemist": ["approval", "objection", "risk_flag", "assay_request"],
}


class MolForgeSharedMixin:
    """Utility methods shared across the environment mixins."""

    def _merge_assays(self, readings: List[AssayReading]) -> None:
        keyed = {
            (reading.tool_name, reading.property_name, reading.molecule_signature): reading
            for reading in self._known_assays
        }
        for reading in readings:
            keyed[(reading.tool_name, reading.property_name, reading.molecule_signature)] = reading
        self._known_assays = list(keyed.values())

    def _current_property_estimate(
        self,
        property_name: str,
        molecule_signature: Optional[str] = None,
    ) -> Optional[float]:
        signature = molecule_signature or self._molecule_signature()
        for reading in reversed(self._known_assays):
            if reading.molecule_signature == signature and reading.property_name == property_name:
                return reading.estimate
        return None

    def _estimate_information_gain(self, tool_name: str) -> float:
        current_signature = self._molecule_signature()
        prior_runs = self._assay_runs.get(f"{current_signature}::{tool_name}", 0)
        base = {
            "evaluate_properties": 0.7,
            "dock_target": 0.62,
            "assay_toxicity": 0.78 if self._scenario.difficulty != "easy" else 0.52,
            "estimate_synthesizability": 0.66 if "synth_min" in self._scenario.hard_constraints else 0.42,
            "evaluate_novelty": 0.38,
            "search_literature": 0.32,
            "run_md_simulation": 0.84,
        }.get(tool_name, 0.25)
        decay = 0.4**prior_runs
        return round(base * decay, 4)

    def _simulate_action_properties(self, action: MolForgeAction) -> Dict[str, float]:
        if action.action_type == "edit" and action.slot:
            molecule = dict(self._molecule)
            if action.edit_type == "remove":
                defaults = {
                    "warhead": "nitrile",
                    "hinge": "pyridine",
                    "solvent_tail": "morpholine",
                    "back_pocket": "methoxy",
                }
                molecule[action.slot] = defaults[action.slot]
            elif action.fragment:
                molecule[action.slot] = action.fragment
            return self._evaluate_for_molecule(molecule, self._trap_penalty_active)

        if action.action_type == "restart":
            return self._evaluate_for_molecule(dict(self._scenario.restart_scaffold), False)

        return self._true_properties()

    def _record_message(self, message: AgentMessage) -> None:
        if not message.message_id:
            message.message_id = f"msg_{self._state.step_count:03d}_{len(self._message_log):03d}"
        self._message_log.append(deepcopy(message))
        self._state.message_count += 1
        self._role_metrics[message.sender]["messages_sent"] += 1
        if message.message_type in {"objection", "risk_flag", "rejection"}:
            self._state.objection_count += 1

    def _sync_state_metadata(self) -> None:
        self._state.metadata = {
            "state_label": self._state.state_label,
            "state_path": list(self._state_path),
            "trace": deepcopy(self._history),
            "message_log": [message.model_dump() for message in self._message_log],
            "oracle_log": deepcopy(self._oracle_log),
            "role_metrics": deepcopy(self._role_metrics),
            "terminal_grader_scores": self._grade_all() if self._state.submitted else {},
        }
        if self._debug_state_enabled:
            self._state.metadata["debug_hidden_properties"] = self._true_properties()

    def _true_properties(self) -> Dict[str, float]:
        return self._evaluate_for_molecule(self._molecule, self._trap_penalty_active)

    def _chemical_diagnostics(self, molecule: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        return molecule_diagnostics(molecule or self._molecule)

    def _evaluate_for_molecule(
        self,
        molecule: Mapping[str, str],
        trap_penalty_active: bool,
    ) -> Dict[str, float]:
        return evaluate_molecule(
            molecule,
            replace(self._scenario, trap_penalty=trap_penalty_active),
            target_shift_active=self._target_shift_active(),
        )

    def _target_shift_active(self) -> bool:
        return bool(
            self._scenario.target_shift_step
            and self._state.step_count >= self._scenario.target_shift_step
        )

    def _molecule_signature(self) -> str:
        return format_molecule(self._molecule)

    def _append_state_label(self, label: str) -> None:
        if not self._state_path or self._state_path[-1] != label:
            self._state_path.append(label)

    def _safety_alerts(self) -> List[str]:
        alerts = []
        if self._molecule["solvent_tail"] == "dimethylamino":
            alerts.append("Dimethylamino tail is a recurring liability for cardiac safety.")
        if self._molecule["back_pocket"] == "trifluoromethyl":
            alerts.append("Trifluoromethyl group may overshoot lipophilic safety windows.")
        if self._molecule["hinge"] == "fluorophenyl" and self._molecule["back_pocket"] == "chloro":
            alerts.append("Hydrophobic hinge/back-pocket combination looks safety-negative.")
        return alerts

    def _route_warnings(self) -> List[str]:
        warnings = []
        if self._molecule["hinge"] == "quinazoline":
            warnings.append("Quinazoline hinge increases route complexity.")
        if self._molecule["warhead"] == "vinyl_sulfonamide":
            warnings.append("Vinyl sulfonamide warhead is reactive and harder to handle.")
        if self._molecule["back_pocket"] == "trifluoromethyl":
            warnings.append("CF3 substitution raises cost and scale-up complexity.")
        return warnings

    @staticmethod
    def _empty_role_metrics() -> Dict[str, Dict[str, int]]:
        return {
            role: {"messages_sent": 0, "correct_messages": 0, "incorrect_messages": 0}
            for role in ["lead_chemist", "toxicologist", "assay_planner", "process_chemist"]
        }

    @staticmethod
    def _open_unit_interval(value: float, epsilon: float = 1e-4) -> float:
        return round(min(max(value, epsilon), 1.0 - epsilon), 4)

    @staticmethod
    def _assay_estimate(
        signature: str,
        tool_name: str,
        property_name: str,
        runs: int,
        true_value: float,
    ) -> float:
        digest = hashlib.sha256(
            f"{signature}|{tool_name}|{property_name}|{runs}".encode("utf-8")
        ).hexdigest()
        centered = (int(digest[:8], 16) / 0xFFFFFFFF) - 0.5
        noise = centered * (0.16 / runs)
        return round(min(max(true_value + noise, 0.0), 1.0), 4)
