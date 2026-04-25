"""Synchronous and async client for the MolForge environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import MolForgeAction, MolForgeObservation, MolForgeState


class MolForgeEnv(EnvClient[MolForgeAction, MolForgeObservation, MolForgeState]):
    """OpenEnv client for the MolForge environment."""

    def _step_payload(self, action: MolForgeAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[MolForgeObservation]:
        obs_data = dict(payload.get("observation", payload))
        obs_data["done"] = payload.get("done", obs_data.get("done", False))
        obs_data["reward"] = payload.get("reward", obs_data.get("reward"))
        observation = MolForgeObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MolForgeState:
        return MolForgeState(**payload)
