"""Organ-module contract and deterministic system assembler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .state import StateRegistry


@dataclass(frozen=True)
class ModelContext:
    """Read-only parameters visible to connected organ modules."""

    patient: Any
    drug: Any
    formulation: Any | None = None


@dataclass(frozen=True)
class ModuleResult:
    """Sparse derivatives and named outgoing connection fluxes."""

    derivatives_mg_h: Mapping[str, float]
    outputs_mg_h: Mapping[str, float] = field(default_factory=dict)


class OrganModule(Protocol):
    """Minimal plug-in interface for one organ or physiological process."""

    name: str
    required_states: tuple[str, ...]
    required_inputs: tuple[str, ...]

    def evaluate(
        self,
        time_h: float,
        amounts_mg: np.ndarray,
        context: ModelContext,
        states: StateRegistry,
        inputs_mg_h: Mapping[str, float],
    ) -> ModuleResult: ...


class SystemAssembler:
    """Compose organ modules into one ODE right-hand side.

    Modules execute in the declared order.  Connection fluxes are immutable
    once published, making accidental double ownership visible immediately.
    State derivatives are additive, which lets circulation terms be owned by
    the organ that generates each exchange.
    """

    def __init__(self, states: StateRegistry, modules: Sequence[OrganModule]) -> None:
        self.states = states
        self.modules = tuple(modules)
        if not self.modules:
            raise ValueError("at least one organ module is required")
        registered = set(states.names)
        for module in self.modules:
            missing = sorted(set(module.required_states) - registered)
            if missing:
                raise ValueError(f"module {module.name} requires unknown states: {missing}")

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        context: ModelContext,
    ) -> np.ndarray:
        raw = np.asarray(state, dtype=np.float64)
        if raw.shape != (len(self.states),):
            raise ValueError(
                f"expected {len(self.states)} model states, got shape {raw.shape}"
            )
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("model state contains non-finite amounts")
        amounts = np.maximum(raw, 0.0)
        derivative = self.states.zeros()
        fluxes: dict[str, float] = {}
        for module in self.modules:
            missing_inputs = sorted(set(module.required_inputs) - set(fluxes))
            if missing_inputs:
                raise RuntimeError(
                    f"module {module.name} is missing connection fluxes: {missing_inputs}"
                )
            result = module.evaluate(time_h, amounts, context, self.states, fluxes)
            for state_name, value in result.derivatives_mg_h.items():
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"module {module.name} produced non-finite d({state_name})/dt"
                    )
                derivative[self.states.index(state_name)] += float(value)
            for port_name, value in result.outputs_mg_h.items():
                if port_name in fluxes:
                    raise RuntimeError(f"connection flux {port_name} has multiple producers")
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"module {module.name} produced non-finite flux {port_name}"
                    )
                fluxes[port_name] = float(value)
        return derivative
