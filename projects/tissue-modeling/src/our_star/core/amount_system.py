"""Species-aware deterministic assembler for micromole PBPK modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .amount_state import AmountStateRegistry, FluxKey


@dataclass(frozen=True)
class AmountModelContext:
    """Read-only subject and parameter context for µmol organ modules."""

    patient: Any
    drug: Any
    formulation: Any | None = None
    mechanism_factors: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factors = {str(key): float(value) for key, value in self.mechanism_factors.items()}
        invalid = {
            key: value
            for key, value in factors.items()
            if not key.strip() or not np.isfinite(value) or value < 0.0
        }
        if invalid:
            raise ValueError(
                "mechanism factors require nonblank names and finite values >= 0: "
                f"{invalid}"
            )
        object.__setattr__(self, "mechanism_factors", MappingProxyType(factors))

    def mechanism_factor(self, mechanism_id: str) -> float:
        """Return a named multiplier, defaulting to no modification."""

        return float(self.mechanism_factors.get(mechanism_id, 1.0))


@dataclass(frozen=True)
class AmountModuleResult:
    """Sparse µmol/h derivatives and species-aware connection outputs."""

    derivatives_umol_h: Mapping[str, float]
    outputs_umol_h: Mapping[FluxKey, float] = field(default_factory=dict)


class AmountOrganModule(Protocol):
    """Minimal contract implemented by one µmol physiological module."""

    name: str
    required_states: tuple[str, ...]
    required_inputs: tuple[FluxKey, ...]

    def evaluate(
        self,
        time_h: float,
        amounts_umol: np.ndarray,
        context: AmountModelContext,
        states: AmountStateRegistry,
        inputs_umol_h: Mapping[FluxKey, float],
    ) -> AmountModuleResult: ...


class AmountSystemAssembler:
    """Compose µmol modules while enforcing flux ownership and finite rates."""

    def __init__(
        self,
        states: AmountStateRegistry,
        modules: Sequence[AmountOrganModule],
    ) -> None:
        self.states = states
        self.modules = tuple(modules)
        if not self.modules:
            raise ValueError("at least one amount organ module is required")
        registered = set(states.names)
        module_names = tuple(module.name for module in self.modules)
        if len(set(module_names)) != len(module_names):
            raise ValueError("amount organ module names must be unique")
        for module in self.modules:
            missing = sorted(set(module.required_states) - registered)
            if missing:
                raise ValueError(
                    f"module {module.name} requires unknown amount states: {missing}"
                )
            if len(set(module.required_inputs)) != len(module.required_inputs):
                raise ValueError(f"module {module.name} repeats a required flux key")

    def rhs(
        self,
        time_h: float,
        state: np.ndarray,
        context: AmountModelContext,
    ) -> np.ndarray:
        raw = self.states.validate_vector(state)
        amounts = np.maximum(raw, 0.0)
        derivative = self.states.zeros()
        fluxes: dict[FluxKey, float] = {}

        for module in self.modules:
            missing_inputs = tuple(
                key for key in module.required_inputs if key not in fluxes
            )
            if missing_inputs:
                raise RuntimeError(
                    f"module {module.name} is missing connection fluxes: "
                    f"{missing_inputs}"
                )
            result = module.evaluate(
                time_h,
                amounts,
                context,
                self.states,
                MappingProxyType(fluxes),
            )
            for state_name, value in result.derivatives_umol_h.items():
                if state_name not in self.states.names:
                    raise KeyError(
                        f"module {module.name} produced derivative for unknown "
                        f"state {state_name}"
                    )
                rate = float(value)
                if not np.isfinite(rate):
                    raise FloatingPointError(
                        f"module {module.name} produced non-finite "
                        f"d({state_name})/dt"
                    )
                derivative[self.states.index(state_name)] += rate

            for raw_key, value in result.outputs_umol_h.items():
                if not isinstance(raw_key, FluxKey):
                    raise TypeError(
                        f"module {module.name} output keys must be FluxKey records"
                    )
                if raw_key in fluxes:
                    raise RuntimeError(
                        f"connection flux {raw_key} has multiple producers"
                    )
                rate = float(value)
                if not np.isfinite(rate) or rate < 0.0:
                    raise FloatingPointError(
                        f"module {module.name} produced invalid non-directional "
                        f"flux {raw_key}: {rate}"
                    )
                fluxes[raw_key] = rate
        return derivative
