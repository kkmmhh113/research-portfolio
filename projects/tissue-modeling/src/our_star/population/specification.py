"""Population specification, constrained transforms, and virtual subjects."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp, expm1, isfinite, log, log1p
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from .random_effects import RandomEffectSpec


ModifierDomain = Literal["anatomy", "mechanism"]
ModifierTransform = Literal["identity", "positive", "log", "logit"]


def _required_name(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _freeze_finite_mapping(
    values: Mapping[str, float], *, label: str
) -> Mapping[str, float]:
    copied: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = _required_name(raw_name, label=f"{label} name")
        if name in copied:
            raise ValueError(f"{label} names must be unique")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{label} values must be finite")
        copied[name] = value
    return MappingProxyType(copied)


def _softplus(value: float) -> float:
    if value > 35.0:
        return value
    if value < -35.0:
        return exp(value)
    return log1p(exp(value))


def _inverse_softplus(value: float) -> float:
    if value > 35.0:
        return value
    return log(expm1(value))


def _logistic(value: float) -> float:
    if value >= 0.0:
        negative = exp(-value)
        return 1.0 / (1.0 + negative)
    positive = exp(value)
    return positive / (1.0 + positive)


def apply_latent_transform(
    reference: float,
    latent_effect: float,
    transform: ModifierTransform,
) -> float:
    """Map an unconstrained latent effect to a physical modifier.

    ``positive`` uses an additive softplus scale and ``log`` uses a
    multiplicative log scale.  Both preserve ``reference`` at a zero latent
    effect but encode different scientific assumptions.  ``logit`` maps to the
    open unit interval and also preserves its reference at zero.
    """

    reference = float(reference)
    latent_effect = float(latent_effect)
    if not isfinite(reference) or not isfinite(latent_effect):
        raise ValueError("reference and latent effect must be finite")
    if transform == "identity":
        result = reference + latent_effect
    elif transform == "positive":
        if reference <= 0.0:
            raise ValueError("positive-transform reference must be > 0")
        result = _softplus(_inverse_softplus(reference) + latent_effect)
    elif transform == "log":
        if reference <= 0.0:
            raise ValueError("log-transform reference must be > 0")
        try:
            result = exp(log(reference) + latent_effect)
        except OverflowError as error:
            raise FloatingPointError(
                "log transform overflowed the physical value"
            ) from error
    elif transform == "logit":
        if not 0.0 < reference < 1.0:
            raise ValueError("logit-transform reference must be in (0, 1)")
        reference_logit = log(reference) - log1p(-reference)
        result = _logistic(reference_logit + latent_effect)
    else:
        raise ValueError(
            "transform must be 'identity', 'positive', 'log', or 'logit'"
        )
    if not isfinite(result):
        raise FloatingPointError("latent transform produced a non-finite value")
    if transform in {"positive", "log"}:
        # Finite log-scale latents can underflow in IEEE-754 arithmetic.  The
        # nearest representable positive value preserves the declared domain
        # without introducing an exact zero into kinetics or anatomy.
        result = max(result, float(np.nextafter(0.0, 1.0)))
    if transform == "logit":
        result = min(
            max(result, float(np.nextafter(0.0, 1.0))),
            float(np.nextafter(1.0, 0.0)),
        )
        if not 0.0 < result < 1.0:
            raise FloatingPointError("logit transform left the open unit interval")
    return result


@dataclass(frozen=True)
class ModifierSpec:
    """Connect one named latent effect to an anatomy or mechanism target."""

    name: str
    target: str
    domain: ModifierDomain
    transform: ModifierTransform
    reference: float
    unit: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_name(self.name, label="modifier name"))
        object.__setattr__(self, "target", _required_name(self.target, label="modifier target"))
        if self.domain not in {"anatomy", "mechanism"}:
            raise ValueError("modifier domain must be 'anatomy' or 'mechanism'")
        if self.transform not in {"identity", "positive", "log", "logit"}:
            raise ValueError(
                "modifier transform must be identity, positive, log, or logit"
            )
        reference = float(self.reference)
        if not np.isfinite(reference):
            raise ValueError("modifier reference must be finite")
        # Calling through the transform validates the reference domain and the
        # zero-latent invariant once, at specification construction time.
        zero_value = apply_latent_transform(reference, 0.0, self.transform)
        if not np.isclose(zero_value, reference, rtol=1e-12, atol=1e-14):
            raise RuntimeError("modifier transform does not preserve its reference")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("modifier unit is required")
        object.__setattr__(self, "reference", reference)

    def apply(self, latent_effect: float) -> float:
        return apply_latent_transform(
            self.reference, float(latent_effect), self.transform
        )


@dataclass(frozen=True)
class FlowCompositionSpec:
    """A cardiac-output composition with an explicit reference component.

    There are ``K - 1`` log-ratio effects for ``K`` flow components.  The last
    component is the reference, avoiding the unidentifiable common-shift
    direction of a K-logit softmax.  Component flows are always composed from
    one total rather than sampled independently.
    """

    total_target: str
    component_targets: tuple[str, ...]
    reference_fractions: tuple[float, ...]
    log_ratio_effect_names: tuple[str, ...]

    def __post_init__(self) -> None:
        total_target = _required_name(self.total_target, label="flow total target")
        component_targets = tuple(self.component_targets)
        if len(component_targets) < 2:
            raise ValueError("flow composition requires at least two components")
        for target in component_targets:
            _required_name(target, label="flow component target")
        if len(set(component_targets)) != len(component_targets):
            raise ValueError("flow component targets must be unique")
        if total_target in component_targets:
            raise ValueError("flow total target must differ from component targets")

        fractions = tuple(float(value) for value in self.reference_fractions)
        if len(fractions) != len(component_targets):
            raise ValueError("one reference fraction is required per flow component")
        if any(not np.isfinite(value) or value <= 0.0 for value in fractions):
            raise ValueError("flow reference fractions must be finite and positive")
        if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("flow reference fractions must sum to one")

        effects = tuple(self.log_ratio_effect_names)
        if len(effects) != len(component_targets) - 1:
            raise ValueError(
                "flow composition requires one log-ratio effect per "
                "non-reference component"
            )
        for effect in effects:
            _required_name(effect, label="flow log-ratio effect name")
        if len(set(effects)) != len(effects):
            raise ValueError("flow log-ratio effect names must be unique")
        object.__setattr__(self, "total_target", total_target)
        object.__setattr__(self, "component_targets", component_targets)
        object.__setattr__(self, "reference_fractions", fractions)
        object.__setattr__(self, "log_ratio_effect_names", effects)

    @property
    def reference_component(self) -> str:
        return self.component_targets[-1]

    def compose(
        self,
        total_flow: float,
        latent_effects: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return component flows whose sum is exactly the supplied total."""

        total = float(total_flow)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("total flow must be finite and positive")
        missing = [
            name for name in self.log_ratio_effect_names if name not in latent_effects
        ]
        if missing:
            raise KeyError(f"missing flow log-ratio effects: {missing}")
        effects = np.asarray(
            [float(latent_effects[name]) for name in self.log_ratio_effect_names],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(effects)):
            raise ValueError("flow log-ratio effects must be finite")

        fractions = np.asarray(self.reference_fractions, dtype=np.float64)
        logits = np.concatenate(
            (
                np.log(fractions[:-1] / fractions[-1]) + effects,
                np.zeros(1, dtype=np.float64),
            )
        )
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        shares = weights / float(weights.sum())
        flows = total * shares
        # Assign the reference component as the residual.  This makes the
        # structural equality exact in float arithmetic, rather than merely
        # close after K independent multiplications.
        subtotal = sum(float(value) for value in flows[:-1])
        flows[-1] = total - subtotal
        if np.any(~np.isfinite(flows)) or np.any(flows <= 0.0):
            raise FloatingPointError(
                "flow composition produced non-positive or non-finite flow"
            )
        result = {
            target: float(value)
            for target, value in zip(self.component_targets, flows, strict=True)
        }
        # Choose the closest representable residual.  In rare cases there is
        # no binary64 value X for which the rounded expression ``S + X`` is
        # bitwise equal to an independently represented total T; the remaining
        # discrepancy is then at most one ULP and is a numerical, not
        # structural, imbalance.
        reference_target = self.reference_component
        original_residual = result[reference_target]
        candidates = (
            original_residual,
            float(np.nextafter(original_residual, -np.inf)),
            float(np.nextafter(original_residual, np.inf)),
        )
        result[reference_target] = min(
            candidates,
            key=lambda candidate: abs(
                total
                - sum(
                    candidate if key == reference_target else value
                    for key, value in result.items()
                )
            ),
        )
        balance_error = abs(total - sum(result.values()))
        balance_tolerance = 2.0 * float(np.spacing(total))
        if (
            balance_error > balance_tolerance
            or result[reference_target] <= 0.0
        ):
            raise RuntimeError(
                "flow composition failed machine-precision structural balance"
            )
        return MappingProxyType(result)


@dataclass(frozen=True)
class PopulationSpecification:
    """Complete mapping from named random effects to virtual-subject fields."""

    population_id: str
    random_effects: RandomEffectSpec
    modifiers: tuple[ModifierSpec, ...]
    flow_composition: FlowCompositionSpec | None = None

    def __post_init__(self) -> None:
        population_id = _required_name(self.population_id, label="population_id")
        if not isinstance(self.random_effects, RandomEffectSpec):
            raise TypeError("random_effects must be a RandomEffectSpec")
        modifiers = tuple(self.modifiers)
        if any(not isinstance(modifier, ModifierSpec) for modifier in modifiers):
            raise TypeError("modifiers must contain only ModifierSpec values")
        modifier_names = tuple(modifier.name for modifier in modifiers)
        modifier_targets = tuple(modifier.target for modifier in modifiers)
        if len(set(modifier_names)) != len(modifier_names):
            raise ValueError("modifier names must be unique")
        if len(set(modifier_targets)) != len(modifier_targets):
            raise ValueError("modifier targets must be unique")

        consumed_names = list(modifier_names)
        if self.flow_composition is not None:
            if not isinstance(self.flow_composition, FlowCompositionSpec):
                raise TypeError("flow_composition must be a FlowCompositionSpec")
            consumed_names.extend(self.flow_composition.log_ratio_effect_names)
            if any(
                target in modifier_targets
                for target in self.flow_composition.component_targets
            ):
                raise ValueError(
                    "composed flow components cannot also be independent modifiers"
                )
            total_matches = [
                modifier
                for modifier in modifiers
                if modifier.target == self.flow_composition.total_target
            ]
            if len(total_matches) != 1 or total_matches[0].domain != "anatomy":
                raise ValueError(
                    "flow total target must be supplied by exactly one anatomy modifier"
                )
        if len(set(consumed_names)) != len(consumed_names):
            raise ValueError("each latent effect must be consumed exactly once")
        if set(consumed_names) != set(self.random_effects.names):
            missing = sorted(set(self.random_effects.names) - set(consumed_names))
            unknown = sorted(set(consumed_names) - set(self.random_effects.names))
            raise ValueError(
                "modifier and flow-effect names must exactly cover random effects; "
                f"missing={missing}, unknown={unknown}"
            )
        object.__setattr__(self, "population_id", population_id)
        object.__setattr__(self, "modifiers", modifiers)

    @property
    def fingerprint(self) -> str:
        flow: dict[str, object] | None = None
        if self.flow_composition is not None:
            flow = {
                "total_target": self.flow_composition.total_target,
                "component_targets": self.flow_composition.component_targets,
                "reference_fractions": self.flow_composition.reference_fractions,
                "log_ratio_effect_names": (
                    self.flow_composition.log_ratio_effect_names
                ),
            }
        payload = {
            "population_id": self.population_id,
            "random_effect_fingerprint": self.random_effects.fingerprint,
            "modifiers": [
                {
                    "name": modifier.name,
                    "target": modifier.target,
                    "domain": modifier.domain,
                    "transform": modifier.transform,
                    "reference": modifier.reference,
                    "unit": modifier.unit,
                }
                for modifier in self.modifiers
            ],
            "flow_composition": flow,
        }
        return sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class VirtualSubject:
    """One immutable virtual subject with anatomy/mechanism separation."""

    subject_id: str
    population_id: str
    draw_index: int
    sampling_seed: int
    latent_effects: Mapping[str, float]
    anatomy_modifiers: Mapping[str, float]
    mechanism_modifiers: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _required_name(self.subject_id, label="subject_id"))
        object.__setattr__(self, "population_id", _required_name(self.population_id, label="population_id"))
        if (
            isinstance(self.draw_index, bool)
            or not isinstance(self.draw_index, (int, np.integer))
            or self.draw_index < 0
        ):
            raise ValueError("draw_index must be a non-negative integer")
        if (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, (int, np.integer))
            or self.sampling_seed < 0
        ):
            raise ValueError("sampling_seed must be a non-negative integer")
        latent = _freeze_finite_mapping(self.latent_effects, label="latent effect")
        anatomy = _freeze_finite_mapping(
            self.anatomy_modifiers, label="anatomy modifier"
        )
        mechanism = _freeze_finite_mapping(
            self.mechanism_modifiers, label="mechanism modifier"
        )
        overlap = sorted(set(anatomy).intersection(mechanism))
        if overlap:
            raise ValueError(
                "anatomy and mechanism target names must not overlap: "
                f"{overlap}"
            )
        object.__setattr__(self, "draw_index", int(self.draw_index))
        object.__setattr__(self, "sampling_seed", int(self.sampling_seed))
        object.__setattr__(self, "latent_effects", latent)
        object.__setattr__(self, "anatomy_modifiers", anatomy)
        object.__setattr__(self, "mechanism_modifiers", mechanism)

    def record(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "population_id": self.population_id,
            "draw_index": self.draw_index,
            "sampling_seed": self.sampling_seed,
            "latent_effects": dict(self.latent_effects),
            "anatomy_modifiers": dict(self.anatomy_modifiers),
            "mechanism_modifiers": dict(self.mechanism_modifiers),
        }


__all__ = [
    "FlowCompositionSpec",
    "ModifierDomain",
    "ModifierSpec",
    "ModifierTransform",
    "PopulationSpecification",
    "VirtualSubject",
    "apply_latent_transform",
]
