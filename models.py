"""Typed models for the MolForge OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


EDIT_TYPES = Literal["add_fragment", "substitute", "remove", "undo_last_edit"]
ACTION_TYPES = Literal["edit", "run_assay", "submit", "restart", "defer"]
TOOL_TYPES = Literal[
    "evaluate_properties",
    "dock_target",
    "assay_toxicity",
    "estimate_synthesizability",
    "evaluate_novelty",
    "search_literature",
    "run_md_simulation",
]
SLOT_TYPES = Literal["warhead", "hinge", "solvent_tail", "back_pocket"]
ROLE_TYPES = Literal[
    "lead_chemist",
    "toxicologist",
    "assay_planner",
    "process_chemist",
    "team",
]
MESSAGE_TYPES = Literal[
    "proposal",
    "objection",
    "risk_flag",
    "assay_request",
    "approval",
    "rejection",
    "revision_request",
    "submission_recommendation",
]
SEVERITY_TYPES = Literal["low", "medium", "high", "critical"]
EFFECT_TYPES = Literal["up", "down", "neutral", "unknown", "not_applicable"]
COORDINATION_MODES = Literal["single_agent", "multi_agent"]
GOVERNANCE_STATES = Literal["ready", "executed", "needs_revision", "policy_veto"]


class MoleculeSlot(BaseModel):
    """Visible fragment assignment for a molecule slot."""

    slot: SLOT_TYPES
    fragment: str = Field(..., description="Selected fragment for the slot")
    editable: bool = Field(default=True, description="Whether the slot is editable")


class AssayReading(BaseModel):
    """Structured oracle result surfaced to the agent."""

    tool_name: str
    property_name: str
    estimate: float = Field(..., ge=0.0, le=1.0)
    confidence_low: float = Field(..., ge=0.0, le=1.0)
    confidence_high: float = Field(..., ge=0.0, le=1.0)
    runs: int = Field(default=1, ge=1)
    molecule_signature: str
    summary: str = ""


class RewardComponent(BaseModel):
    """Named reward component used in report cards and debugging."""

    name: str
    value: float
    explanation: str


class ConstraintCheck(BaseModel):
    """Constraint status based only on currently visible evidence."""

    name: str
    target: str
    satisfied: Optional[bool] = None
    actual: Optional[float] = None
    evidence_status: Literal["known", "unknown"] = "unknown"


class AgentMessage(BaseModel):
    """Structured inter-agent communication message."""

    message_id: str = ""
    sender: ROLE_TYPES
    receiver: str = "team"
    message_type: MESSAGE_TYPES
    severity: SEVERITY_TYPES = "low"
    reference_action_type: Optional[ACTION_TYPES] = None
    summary: str = Field(default="", max_length=240)
    payload: Dict[str, Any] = Field(default_factory=dict)


class RoleObservation(BaseModel):
    """Role-specific structured observation slice."""

    role: ROLE_TYPES
    local_objective: str
    permissions: List[str] = Field(default_factory=list)
    observation: Dict[str, Any] = Field(default_factory=dict)


class GovernanceStatus(BaseModel):
    """Outcome of the multi-agent review process for the last turn."""

    status: GOVERNANCE_STATES = "ready"
    explanation: str = ""
    required_roles: List[str] = Field(default_factory=list)
    approvals: List[str] = Field(default_factory=list)
    objections: List[str] = Field(default_factory=list)
    vetoes: List[str] = Field(default_factory=list)
    executable: bool = True


class MolForgeAction(Action):
    """Single team turn action spanning edits, assays, messages, and submission."""

    action_type: ACTION_TYPES = Field(
        ..., description="High-level action type to execute this turn"
    )
    acting_role: ROLE_TYPES = Field(
        default="lead_chemist",
        description="Role claiming ownership of the executable team decision",
    )
    edit_type: Optional[EDIT_TYPES] = Field(
        default=None, description="Edit subtype when action_type is edit"
    )
    slot: Optional[SLOT_TYPES] = Field(
        default=None, description="Editable molecular slot when performing edits"
    )
    fragment: Optional[str] = Field(
        default=None, description="Fragment identifier for edit actions"
    )
    tool_name: Optional[TOOL_TYPES] = Field(
        default=None, description="Oracle or tool name for run_assay actions"
    )
    messages: List[AgentMessage] = Field(
        default_factory=list,
        description="Structured multi-agent communication bundle for this decision turn",
    )
    rationale: str = Field(
        default="",
        description="Short explanation of why the final decision should help",
        max_length=400,
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Visible observation facts supporting the action; do not include hidden state.",
        max_length=5,
    )
    expected_effects: Dict[str, EFFECT_TYPES] = Field(
        default_factory=dict,
        description="Directional public prediction for potency, toxicity, synth, novelty, or budget.",
    )


class MolForgeObservation(Observation):
    """Observation emitted after reset and each step."""

    scenario_id: str
    difficulty: str
    state_label: str = "[start]"
    state_path: List[str] = Field(default_factory=list)
    coordination_mode: COORDINATION_MODES = "multi_agent"
    enabled_roles: List[str] = Field(default_factory=list)
    task_brief: str
    target_name: str
    current_molecule: str
    molecule_slots: List[MoleculeSlot] = Field(default_factory=list)
    editable_slots: List[str] = Field(default_factory=list)
    step_index: int = Field(default=0, ge=0)
    max_steps: int = Field(default=0, ge=1)
    remaining_budget: int = Field(default=0, ge=0)
    budget_used: int = Field(default=0, ge=0)
    max_budget: int = Field(default=0, ge=1)
    known_assays: List[AssayReading] = Field(default_factory=list)
    role_observations: List[RoleObservation] = Field(default_factory=list)
    message_log: List[AgentMessage] = Field(default_factory=list)
    governance: GovernanceStatus = Field(default_factory=GovernanceStatus)
    last_transition_summary: str = ""
    visible_metrics: Dict[str, float] = Field(default_factory=dict)
    constraint_status: List[ConstraintCheck] = Field(default_factory=list)
    reward_breakdown: List[RewardComponent] = Field(default_factory=list)
    allowed_actions: List[str] = Field(default_factory=list)
    report_card: str = ""


class MolForgeState(State):
    """Internal environment state surfaced through the state() API."""

    scenario_id: str = ""
    difficulty: str = ""
    state_label: str = "[start]"
    state_path: List[str] = Field(default_factory=list)
    coordination_mode: COORDINATION_MODES = "multi_agent"
    enabled_roles: List[str] = Field(default_factory=list)
    target_name: str = ""
    current_molecule: str = ""
    remaining_budget: int = 0
    budget_used: int = 0
    max_budget: int = 0
    visited_states: int = 0
    known_assay_count: int = 0
    invalid_action_count: int = 0
    objection_count: int = 0
    oracle_call_count: int = 0
    message_count: int = 0
    decision_count: int = 0
    submitted: bool = False
    last_error_code: str = ""
    reward_total: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
