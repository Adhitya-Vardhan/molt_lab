# -*- coding: utf-8 -*-
"""Lightweight openenv-core shim for environments that only need the base types.

Import this module **before** any ``from openenv.core...`` imports when the
full ``openenv-core`` package is not installed (e.g. Colab RL training).  It
registers minimal stubs into ``sys.modules`` so that the following imports
work identically to the real package:

    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.env_server.interfaces import Environment

Usage::

    try:
        import openenv          # real package available
    except ImportError:
        import openenv_shim     # registers lightweight stubs
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ── Base types (mirror openenv.core.env_server.types) ────────────────────

class Action(BaseModel):
    """Minimal action base matching openenv-core's Action."""

    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Minimal observation base matching openenv-core's Observation."""

    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Minimal state base matching openenv-core's State."""

    episode_id: str = ""
    step_count: int = 0


# ── Environment ABC (mirror openenv.core.env_server.interfaces) ──────────

class Environment(ABC):
    """Minimal environment ABC matching openenv-core's Environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, **_kwargs: Any):
        pass

    @abstractmethod
    def reset(self, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def step(self, action: Any, **kwargs: Any) -> Any:
        ...

    @property
    @abstractmethod
    def state(self) -> Any:
        ...


# ── Register shim modules into sys.modules ───────────────────────────────

def _register() -> None:
    """Inject stub modules so ``from openenv.core...`` imports resolve."""

    # Build the types module
    types_mod = ModuleType("openenv.core.env_server.types")
    types_mod.Action = Action  # type: ignore[attr-defined]
    types_mod.Observation = Observation  # type: ignore[attr-defined]
    types_mod.State = State  # type: ignore[attr-defined]

    # Build the interfaces module
    interfaces_mod = ModuleType("openenv.core.env_server.interfaces")
    interfaces_mod.Environment = Environment  # type: ignore[attr-defined]

    # Build the package hierarchy
    openenv_mod = ModuleType("openenv")
    core_mod = ModuleType("openenv.core")
    env_server_mod = ModuleType("openenv.core.env_server")

    # Wire up sub-modules
    env_server_mod.types = types_mod  # type: ignore[attr-defined]
    env_server_mod.interfaces = interfaces_mod  # type: ignore[attr-defined]
    core_mod.env_server = env_server_mod  # type: ignore[attr-defined]
    openenv_mod.core = core_mod  # type: ignore[attr-defined]

    # Register everything
    for name, mod in [
        ("openenv", openenv_mod),
        ("openenv.core", core_mod),
        ("openenv.core.env_server", env_server_mod),
        ("openenv.core.env_server.types", types_mod),
        ("openenv.core.env_server.interfaces", interfaces_mod),
    ]:
        sys.modules.setdefault(name, mod)


_register()
