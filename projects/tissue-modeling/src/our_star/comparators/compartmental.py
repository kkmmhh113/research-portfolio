"""Low-dimensional empirical compartmental PK comparators.

The oral one-compartment model is deliberately an apparent-parameter model::

    dAg/dt = -ka * Ag
    dAc/dt =  ka * Ag - ke * Ac
    C(t)   = scale * Ac

For an oral aggregate concentration curve, ``scale_l_inv`` represents ``F/V``.
Bioavailability and volume are not separately identified and must not be
interpreted from this model.  The implementation is research-only and contains
no clinical-data loading or fitting behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ORAL_ONE_COMPARTMENT_PARAMETER_NAMES = (
    "ka_h",
    "ke_h",
    "scale_l_inv",
)


def _positive_finite(value: float, *, label: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return converted


@dataclass(frozen=True)
class OralOneCompartmentParameters:
    """Positive apparent parameters for the oral one-compartment model."""

    ka_h: float
    ke_h: float
    scale_l_inv: float

    def __post_init__(self) -> None:
        for name in ORAL_ONE_COMPARTMENT_PARAMETER_NAMES:
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), label=name),
            )

    def as_array(self) -> np.ndarray:
        """Return parameters in the declared, stable model order."""

        return np.asarray(
            [self.ka_h, self.ke_h, self.scale_l_inv], dtype=np.float64
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> OralOneCompartmentParameters:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(ORAL_ONE_COMPARTMENT_PARAMETER_NAMES),):
            raise ValueError("oral one-compartment values must have shape (3,)")
        return cls(*map(float, array))

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(
                ORAL_ONE_COMPARTMENT_PARAMETER_NAMES,
                self.as_array(),
                strict=True,
            )
        }


@dataclass(frozen=True)
class OralOneCompartmentBounds:
    """Explicit finite bounds for every fitted positive parameter.

    Each field is ``(lower, upper)`` in physical units.  The optimizer maps
    these bounds to log-parameter coordinates; the bounds themselves remain
    human-readable for protocol freezing and audit reports.
    """

    ka_h: tuple[float, float]
    ke_h: tuple[float, float]
    scale_l_inv: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ORAL_ONE_COMPARTMENT_PARAMETER_NAMES:
            raw = getattr(self, name)
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise TypeError(f"{name} bounds must be a (lower, upper) pair")
            lower = _positive_finite(raw[0], label=f"{name} lower bound")
            upper = _positive_finite(raw[1], label=f"{name} upper bound")
            if lower >= upper:
                raise ValueError(f"{name} bounds must satisfy lower < upper")
            object.__setattr__(self, name, (lower, upper))

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name)[0] for name in ORAL_ONE_COMPARTMENT_PARAMETER_NAMES],
            dtype=np.float64,
        )

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name)[1] for name in ORAL_ONE_COMPARTMENT_PARAMETER_NAMES],
            dtype=np.float64,
        )

    @property
    def log_lower(self) -> np.ndarray:
        return np.log(self.lower)

    @property
    def log_upper(self) -> np.ndarray:
        return np.log(self.upper)

    def validate(self, parameters: OralOneCompartmentParameters) -> None:
        if not isinstance(parameters, OralOneCompartmentParameters):
            raise TypeError("parameters must be OralOneCompartmentParameters")
        values = parameters.as_array()
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError("oral one-compartment parameters lie outside bounds")

    def to_dict(self) -> dict[str, list[float]]:
        return {
            name: [float(lower), float(upper)]
            for name in ORAL_ONE_COMPARTMENT_PARAMETER_NAMES
            for lower, upper in (getattr(self, name),)
        }


def oral_one_compartment_concentration(
    times_h: np.ndarray,
    *,
    dose_mg: float,
    parameters: OralOneCompartmentParameters,
) -> np.ndarray:
    """Evaluate the exact oral one-compartment concentration-time solution.

    ``Ag(0)`` equals ``dose_mg`` and ``Ac(0)`` equals zero.  A numerically
    stable ``expm1`` expression is used when ``ka_h`` and ``ke_h`` are close;
    the exact repeated-rate limit is used when they are indistinguishable at
    floating-point precision.
    """

    if not isinstance(parameters, OralOneCompartmentParameters):
        raise TypeError("parameters must be OralOneCompartmentParameters")
    dose = _positive_finite(dose_mg, label="dose_mg")
    times = np.asarray(times_h, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_h must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times_h must contain only finite, nonnegative values")

    ka_h = parameters.ka_h
    ke_h = parameters.ke_h
    difference = ka_h - ke_h
    rate_scale = max(ka_h, ke_h)
    if abs(difference) <= 32.0 * np.finfo(np.float64).eps * rate_scale:
        central_amount_mg = dose * ka_h * times * np.exp(-ka_h * times)
    elif difference > 0.0:
        central_amount_mg = (
            dose
            * ka_h
            * np.exp(-ke_h * times)
            * (-np.expm1(-difference * times))
            / difference
        )
    else:
        # Factoring out exp(-ka*t) avoids the overflow-times-underflow form
        # that would arise from exp(-ke*t) * exp((ke-ka)*t) when ke > ka.
        central_amount_mg = (
            dose
            * ka_h
            * np.exp(-ka_h * times)
            * np.expm1(difference * times)
            / difference
        )
    concentration = parameters.scale_l_inv * central_amount_mg
    # Roundoff can only create vanishingly small negatives in the difference
    # expression; materially negative values indicate an implementation error.
    tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(concentration))
    )
    if np.any(concentration < -tolerance) or np.any(~np.isfinite(concentration)):
        raise FloatingPointError("oral one-compartment solution became invalid")
    return np.maximum(concentration, 0.0)
